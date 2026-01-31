import contextlib
import dataclasses
import unittest
from collections import defaultdict
from typing import DefaultDict, Dict
import torch
from src.AutoEncoders.ResBlock import forward_timestep_embed1
from src.NeuralNetwork.unet import apply_control1
from src.sample.sampling_util import timestep_embedding

_current_cache_context = None


@dataclasses.dataclass
class CacheContext:
    buffers: Dict[str, torch.Tensor] = dataclasses.field(default_factory=dict)
    incremental_name_counters: DefaultDict[str, int] = dataclasses.field(default_factory=lambda: defaultdict(int))

    def get_incremental_name(self, name=None):
        name = name or "default"
        idx = self.incremental_name_counters[name]
        self.incremental_name_counters[name] += 1
        return f"{name}_{idx}"

    def reset_incremental_names(self):
        self.incremental_name_counters.clear()

    @torch.compiler.disable()
    def get_buffer(self, name):
        return self.buffers.get(name)

    @torch.compiler.disable()
    def set_buffer(self, name, buffer):
        self.buffers[name] = buffer

    def clear_buffers(self):
        self.buffers.clear()


def create_cache_context():
    return CacheContext()


def get_current_cache_context():
    return _current_cache_context


def set_current_cache_context(cache_context=None):
    global _current_cache_context
    _current_cache_context = cache_context


@contextlib.contextmanager
def cache_context(ctx):
    global _current_cache_context
    old = _current_cache_context
    _current_cache_context = ctx
    try:
        yield
    finally:
        _current_cache_context = old


@torch.compiler.disable()
def get_buffer(name):
    ctx = get_current_cache_context()
    assert ctx is not None
    return ctx.get_buffer(name)


@torch.compiler.disable()
def set_buffer(name, buffer):
    ctx = get_current_cache_context()
    assert ctx is not None
    ctx.set_buffer(name, buffer)


@torch.compiler.disable()
def are_two_tensors_similar(t1, t2, *, threshold):
    if t1.shape != t2.shape:
        return False
    return ((t1 - t2).abs().mean() / t1.abs().mean()).item() < threshold


@torch.compiler.disable()
def apply_prev_hidden_states_residual(hidden_states, encoder_hidden_states=None):
    hidden_states = (get_buffer("hidden_states_residual") + hidden_states).contiguous()
    if encoder_hidden_states is None:
        return hidden_states
    enc_res = get_buffer("encoder_hidden_states_residual")
    if enc_res is None:
        return hidden_states, None
    return hidden_states, (enc_res + encoder_hidden_states).contiguous()


@torch.compiler.disable()
def get_can_use_cache(first_hidden_states_residual, threshold, parallelized=False):
    prev = get_buffer("first_hidden_states_residual")
    return prev is not None and are_two_tensors_similar(prev, first_hidden_states_residual, threshold=threshold)


class CachedTransformerBlocks(torch.nn.Module):
    def __init__(self, transformer_blocks, single_transformer_blocks=None, *, residual_diff_threshold,
                 validate_can_use_cache_function=None, return_hidden_states_first=True,
                 accept_hidden_states_first=True, cat_hidden_states_first=False,
                 return_hidden_states_only=False, clone_original_hidden_states=False):
        super().__init__()
        self.transformer_blocks = transformer_blocks
        self.single_transformer_blocks = single_transformer_blocks
        self.residual_diff_threshold = residual_diff_threshold
        self.validate_can_use_cache_function = validate_can_use_cache_function
        self.return_hidden_states_first = return_hidden_states_first
        self.accept_hidden_states_first = accept_hidden_states_first
        self.cat_hidden_states_first = cat_hidden_states_first
        self.return_hidden_states_only = return_hidden_states_only
        self.clone_original_hidden_states = clone_original_hidden_states

    def _extract_args(self, args, kwargs):
        img_key = "img" if "img" in kwargs else "hidden_states" if "hidden_states" in kwargs else None
        txt_key = "txt" if "txt" in kwargs else "context" if "context" in kwargs else "encoder_hidden_states" if "encoder_hidden_states" in kwargs else None
        args = list(args)
        if self.accept_hidden_states_first:
            img = args.pop(0) if args else kwargs.pop(img_key)
            txt = args.pop(0) if args else kwargs.pop(txt_key)
        else:
            txt = args.pop(0) if args else kwargs.pop(txt_key)
            img = args.pop(0) if args else kwargs.pop(img_key)
        return img, txt, txt_key, args, kwargs

    def _call_block(self, block, img, txt, txt_key, args, kwargs):
        if txt_key == "encoder_hidden_states":
            out = block(img, *args, encoder_hidden_states=txt, **kwargs)
        elif self.accept_hidden_states_first:
            out = block(img, txt, *args, **kwargs)
        else:
            out = block(txt, img, *args, **kwargs)
        if not self.return_hidden_states_only:
            img, txt = out
            if not self.return_hidden_states_first:
                img, txt = txt, img
        else:
            img = out
        return img, txt

    def _process_single_blocks(self, img, txt, args, kwargs):
        if self.single_transformer_blocks is None:
            return img, txt
        img = torch.cat([img, txt] if self.cat_hidden_states_first else [txt, img], dim=1)
        for block in self.single_transformer_blocks:
            img = block(img, *args, **kwargs)
        return img[:, txt.shape[1]:] if self.cat_hidden_states_first else img[:, txt.shape[1]:], txt

    def _format_output(self, img, txt):
        if self.return_hidden_states_only:
            return img
        return (img, txt) if self.return_hidden_states_first else (txt, img)

    def forward(self, *args, **kwargs):
        img, txt, txt_key, args, kwargs = self._extract_args(args, kwargs)
        if self.residual_diff_threshold <= 0.0:
            for block in self.transformer_blocks:
                img, txt = self._call_block(block, img, txt, txt_key, args, kwargs)
            img, txt = self._process_single_blocks(img, txt, args, kwargs)
            return self._format_output(img, txt)

        original_img = img.clone() if self.clone_original_hidden_states else img
        img, txt = self._call_block(self.transformer_blocks[0], img, txt, txt_key, args, kwargs)
        first_residual = img - original_img

        can_use_cache = get_can_use_cache(first_residual, threshold=self.residual_diff_threshold)
        if self.validate_can_use_cache_function:
            can_use_cache = self.validate_can_use_cache_function(can_use_cache)

        torch._dynamo.graph_break()
        if can_use_cache:
            result = apply_prev_hidden_states_residual(img, txt)
            img, txt = (result, txt) if isinstance(result, torch.Tensor) else result
        else:
            set_buffer("first_hidden_states_residual", first_residual)
            img, txt, img_res, txt_res = self._call_remaining(img, txt, txt_key, args, kwargs)
            set_buffer("hidden_states_residual", img_res)
            if txt_res is not None:
                set_buffer("encoder_hidden_states_residual", txt_res)
        torch._dynamo.graph_break()
        return self._format_output(img, txt)

    def _call_remaining(self, img, txt, txt_key, args, kwargs):
        orig_img = img.clone() if self.clone_original_hidden_states else img
        orig_txt = txt.clone() if self.clone_original_hidden_states and txt is not None else txt
        for block in self.transformer_blocks[1:]:
            img, txt = self._call_block(block, img, txt, txt_key, args, kwargs)
        if self.single_transformer_blocks:
            img = torch.cat([img, txt] if self.cat_hidden_states_first else [txt, img], dim=1)
            for block in self.single_transformer_blocks:
                img = block(img, *args, **kwargs)
            if self.cat_hidden_states_first:
                img, txt = img.split([img.shape[1] - txt.shape[1], txt.shape[1]], dim=1)
            else:
                txt, img = img.split([txt.shape[1], img.shape[1] - txt.shape[1]], dim=1)
        img = img.flatten().contiguous().reshape(img.shape)
        if txt is not None:
            txt = txt.flatten().contiguous().reshape(txt.shape)
        return img, txt, img - orig_img, (txt - orig_txt if txt is not None else None)


def create_patch_unet_model__forward(model, *, residual_diff_threshold, validate_can_use_cache_function=None):
    def call_remaining_blocks(self, transformer_options, control, transformer_patches, hs, h, *args, **kwargs):
        original_h = h
        for id, module in enumerate(self.input_blocks):
            if id < 2:
                continue
            transformer_options["block"] = ("input", id)
            h = forward_timestep_embed1(module, h, *args, **kwargs)
            h = apply_control1(h, control, 'input')
            for p in transformer_patches.get("input_block_patch", []):
                h = p(h, transformer_options)
            hs.append(h)
            for p in transformer_patches.get("input_block_patch_after_skip", []):
                h = p(h, transformer_options)

        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = forward_timestep_embed1(self.middle_block, h, *args, **kwargs)
        h = apply_control1(h, control, 'middle')

        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = apply_control1(hs.pop(), control, 'output')
            for p in transformer_patches.get("output_block_patch", []):
                h, hsp = p(h, hsp, transformer_options)
            h = torch.cat([h, hsp], dim=1)
            del hsp
            h = forward_timestep_embed1(module, h, *args, hs[-1].shape if hs else None, **kwargs)
        return h, h - original_h

    def unet_forward(self, x, timesteps=None, context=None, y=None, control=None, transformer_options={}, **kwargs):
        transformer_options["original_shape"], transformer_options["transformer_index"] = list(x.shape), 0
        transformer_patches = transformer_options.get("patches", {})
        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator, time_context = kwargs.get("image_only_indicator"), kwargs.get("time_context")
        assert (y is not None) == (self.num_classes is not None)

        emb = self.time_embed(timestep_embedding(timesteps, self.model_channels, repeat_only=False).to(x.dtype))
        for p in transformer_patches.get("emb_patch", []):
            emb = p(emb, self.model_channels, transformer_options)
        if self.num_classes is not None:
            emb = emb + self.label_emb(y)

        hs, h = [], x
        for id, module in enumerate(self.input_blocks):
            if id >= 2:
                break
            transformer_options["block"] = ("input", id)
            if id == 1:
                original_h = h
            h = forward_timestep_embed1(module, h, emb, context, transformer_options, time_context=time_context,
                                       num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control1(h, control, 'input')
            for p in transformer_patches.get("input_block_patch", []):
                h = p(h, transformer_options)
            hs.append(h)
            for p in transformer_patches.get("input_block_patch_after_skip", []):
                h = p(h, transformer_options)
            if id == 1:
                first_residual = h - original_h
                can_use_cache = get_can_use_cache(first_residual, threshold=residual_diff_threshold)
                if validate_can_use_cache_function:
                    can_use_cache = validate_can_use_cache_function(can_use_cache)
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual", first_residual)

        torch._dynamo.graph_break()
        if can_use_cache:
            h = apply_prev_hidden_states_residual(h)
        else:
            h, hidden_states_residual = call_remaining_blocks(self, transformer_options, control, transformer_patches, hs, h,
                emb, context, transformer_options, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            set_buffer("hidden_states_residual", hidden_states_residual)
        torch._dynamo.graph_break()

        return self.id_predictor(h) if self.predict_codebook_ids else self.out(h.type(x.dtype))

    new_forward = unet_forward.__get__(model)

    @contextlib.contextmanager
    def patch__forward():
        with unittest.mock.patch.object(model, "_forward", new_forward):
            yield
    return patch__forward


def create_patch_flux_forward_orig(model, *, residual_diff_threshold, validate_can_use_cache_function=None):
    def call_remaining_blocks(self, blocks_replace, control, img, txt, vec, pe, attn_mask, ca_idx, timesteps, transformer_options):
        original_img = img
        extra_kwargs = {"attn_mask": attn_mask} if attn_mask is not None else {}

        for i, block in enumerate(self.double_blocks):
            if i < 1:
                continue
            if ("double_block", i) in blocks_replace:
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": lambda args: {"img": block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], **extra_kwargs)[0], "txt": block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], **extra_kwargs)[1]}, "transformer_options": transformer_options})
                img, txt = out["img"], out["txt"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_kwargs)
            if control and i < len(control.get("input", [])) and control["input"][i] is not None:
                img += control["input"][i]
            if getattr(self, "pulid_data", {}) and i % self.pulid_double_interval == 0:
                for _, node_data in self.pulid_data.items():
                    if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                        img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                ca_idx += 1

        img = torch.cat((txt, img), 1)
        for i, block in enumerate(self.single_blocks):
            if ("single_block", i) in blocks_replace:
                out = blocks_replace[("single_block", i)]({"img": img, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": lambda args: {"img": block(args["img"], vec=args["vec"], pe=args["pe"], **extra_kwargs)}, "transformer_options": transformer_options})
                img = out["img"]
            else:
                img = block(img, vec=vec, pe=pe, **extra_kwargs)
            if control and i < len(control.get("output", [])) and control["output"][i] is not None:
                img[:, txt.shape[1]:, ...] += control["output"][i]
            if getattr(self, "pulid_data", {}) and i % self.pulid_single_interval == 0:
                real_img, txt_part = img[:, txt.shape[1]:, ...], img[:, :txt.shape[1], ...]
                for _, node_data in self.pulid_data.items():
                    if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                        real_img = real_img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], real_img)
                ca_idx += 1
                img = torch.cat((txt_part, real_img), 1)

        img = img[:, txt.shape[1]:, ...].contiguous()
        return img, img - original_img

    def forward_orig(self, img, img_ids, txt, txt_ids, timesteps, y, guidance=None, control=None, transformer_options={}, attn_mask=None):
        patches_replace = transformer_options.get("patches_replace", {})
        if img.ndim != 3 or txt.ndim != 3:
            raise ValueError("Input tensors must have 3 dimensions.")

        img = self.img_in(img)
        vec = self.time_in(timestep_embedding(timesteps, 256).to(img.dtype))
        if self.params.guidance_embed:
            if guidance is None:
                raise ValueError("Missing guidance for guidance distilled model.")
            vec = vec + self.guidance_in(timestep_embedding(guidance, 256).to(img.dtype))
        vec = vec + self.vector_in(y[:, :self.params.vec_in_dim])
        txt = self.txt_in(txt)
        pe = self.pe_embedder(torch.cat((txt_ids, img_ids), dim=1))

        ca_idx = 0
        extra_kwargs = {"attn_mask": attn_mask} if attn_mask is not None else {}
        blocks_replace = patches_replace.get("dit", {})

        for i, block in enumerate(self.double_blocks):
            if i >= 1:
                break
            if ("double_block", i) in blocks_replace:
                out = blocks_replace[("double_block", i)]({"img": img, "txt": txt, "vec": vec, "pe": pe, **extra_kwargs},
                    {"original_block": lambda args: {"img": block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], **extra_kwargs)[0], "txt": block(img=args["img"], txt=args["txt"], vec=args["vec"], pe=args["pe"], **extra_kwargs)[1]}, "transformer_options": transformer_options})
                img, txt = out["img"], out["txt"]
            else:
                img, txt = block(img=img, txt=txt, vec=vec, pe=pe, **extra_kwargs)
            if control and i < len(control.get("input", [])) and control["input"][i] is not None:
                img += control["input"][i]
            if getattr(self, "pulid_data", {}) and i % self.pulid_double_interval == 0:
                for _, node_data in self.pulid_data.items():
                    if torch.any((node_data['sigma_start'] >= timesteps) & (timesteps >= node_data['sigma_end'])):
                        img = img + node_data['weight'] * self.pulid_ca[ca_idx](node_data['embedding'], img)
                ca_idx += 1

            if i == 0:
                first_residual = img
                can_use_cache = get_can_use_cache(first_residual, threshold=residual_diff_threshold)
                if validate_can_use_cache_function:
                    can_use_cache = validate_can_use_cache_function(can_use_cache)
                if not can_use_cache:
                    set_buffer("first_hidden_states_residual", first_residual)

        torch._dynamo.graph_break()
        if can_use_cache:
            img = apply_prev_hidden_states_residual(img)
        else:
            img, residual = call_remaining_blocks(self, blocks_replace, control, img, txt, vec, pe, attn_mask, ca_idx, timesteps, transformer_options)
            set_buffer("hidden_states_residual", residual)
        torch._dynamo.graph_break()
        return self.final_layer(img, vec)

    new_forward = forward_orig.__get__(model)

    @contextlib.contextmanager
    def patch_forward():
        with unittest.mock.patch.object(model, "forward_orig", new_forward):
            yield
    return patch_forward
