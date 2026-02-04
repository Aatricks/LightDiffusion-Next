import logging
import math
from typing import Any, Dict, List, Optional
import torch.nn as nn
import torch

from src.Utilities import util
from src.AutoEncoders import ResBlock
from src.NeuralNetwork import transformer
from src.cond import cast
from src.sample import sampling, sampling_util

UNET_MAP_ATTENTIONS = {"proj_in.weight", "proj_in.bias", "proj_out.weight", "proj_out.bias", "norm.weight", "norm.bias"}
TRANSFORMER_BLOCKS = {
    "norm1.weight", "norm1.bias", "norm2.weight", "norm2.bias", "norm3.weight", "norm3.bias",
    "attn1.to_q.weight", "attn1.to_k.weight", "attn1.to_v.weight", "attn1.to_out.0.weight", "attn1.to_out.0.bias",
    "attn2.to_q.weight", "attn2.to_k.weight", "attn2.to_v.weight", "attn2.to_out.0.weight", "attn2.to_out.0.bias",
    "ff.net.0.proj.weight", "ff.net.0.proj.bias", "ff.net.2.weight", "ff.net.2.bias",
}
UNET_MAP_RESNET = {
    "in_layers.2.weight": "conv1.weight", "in_layers.2.bias": "conv1.bias",
    "emb_layers.1.weight": "time_emb_proj.weight", "emb_layers.1.bias": "time_emb_proj.bias",
    "out_layers.3.weight": "conv2.weight", "out_layers.3.bias": "conv2.bias",
    "skip_connection.weight": "conv_shortcut.weight", "skip_connection.bias": "conv_shortcut.bias",
    "in_layers.0.weight": "norm1.weight", "in_layers.0.bias": "norm1.bias",
    "out_layers.0.weight": "norm2.weight", "out_layers.0.bias": "norm2.bias",
}
UNET_MAP_BASIC = {
    ("label_emb.0.0.weight", "class_embedding.linear_1.weight"), ("label_emb.0.0.bias", "class_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "class_embedding.linear_2.weight"), ("label_emb.0.2.bias", "class_embedding.linear_2.bias"),
    ("label_emb.0.0.weight", "add_embedding.linear_1.weight"), ("label_emb.0.0.bias", "add_embedding.linear_1.bias"),
    ("label_emb.0.2.weight", "add_embedding.linear_2.weight"), ("label_emb.0.2.bias", "add_embedding.linear_2.bias"),
    ("input_blocks.0.0.weight", "conv_in.weight"), ("input_blocks.0.0.bias", "conv_in.bias"),
    ("out.0.weight", "conv_norm_out.weight"), ("out.0.bias", "conv_norm_out.bias"),
    ("out.2.weight", "conv_out.weight"), ("out.2.bias", "conv_out.bias"),
    ("time_embed.0.weight", "time_embedding.linear_1.weight"), ("time_embed.0.bias", "time_embedding.linear_1.bias"),
    ("time_embed.2.weight", "time_embedding.linear_2.weight"), ("time_embed.2.bias", "time_embedding.linear_2.bias"),
}

oai_ops = cast.disable_weight_init


def unet_to_diffusers(unet_config: dict) -> dict:
    if "num_res_blocks" not in unet_config:
        return {}
    num_res_blocks, channel_mult = unet_config["num_res_blocks"], unet_config["channel_mult"]
    transformer_depth, transformer_depth_output = unet_config["transformer_depth"][:], unet_config["transformer_depth_output"][:]
    num_blocks, transformers_mid = len(channel_mult), unet_config.get("transformer_depth_middle", None)
    diffusers_unet_map = {}

    for x in range(num_blocks):
        n = 1 + (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x]):
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[f"down_blocks.{x}.resnets.{i}.{UNET_MAP_RESNET[b]}"] = f"input_blocks.{n}.0.{b}"
            num_transformers = transformer_depth.pop(0)
            if num_transformers > 0:
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[f"down_blocks.{x}.attentions.{i}.{b}"] = f"input_blocks.{n}.1.{b}"
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[f"down_blocks.{x}.attentions.{i}.transformer_blocks.{t}.{b}"] = f"input_blocks.{n}.1.transformer_blocks.{t}.{b}"
            n += 1
        for k in ["weight", "bias"]:
            diffusers_unet_map[f"down_blocks.{x}.downsamplers.0.conv.{k}"] = f"input_blocks.{n}.0.op.{k}"

    for b in UNET_MAP_ATTENTIONS:
        diffusers_unet_map[f"mid_block.attentions.0.{b}"] = f"middle_block.1.{b}"
    for t in range(transformers_mid):
        for b in TRANSFORMER_BLOCKS:
            diffusers_unet_map[f"mid_block.attentions.0.transformer_blocks.{t}.{b}"] = f"middle_block.1.transformer_blocks.{t}.{b}"
    for i, n in enumerate([0, 2]):
        for b in UNET_MAP_RESNET:
            diffusers_unet_map[f"mid_block.resnets.{i}.{UNET_MAP_RESNET[b]}"] = f"middle_block.{n}.{b}"

    num_res_blocks = list(reversed(num_res_blocks))
    for x in range(num_blocks):
        n = (num_res_blocks[x] + 1) * x
        for i in range(num_res_blocks[x] + 1):
            c = 0
            for b in UNET_MAP_RESNET:
                diffusers_unet_map[f"up_blocks.{x}.resnets.{i}.{UNET_MAP_RESNET[b]}"] = f"output_blocks.{n}.0.{b}"
            c += 1
            num_transformers = transformer_depth_output.pop()
            if num_transformers > 0:
                c += 1
                for b in UNET_MAP_ATTENTIONS:
                    diffusers_unet_map[f"up_blocks.{x}.attentions.{i}.{b}"] = f"output_blocks.{n}.1.{b}"
                for t in range(num_transformers):
                    for b in TRANSFORMER_BLOCKS:
                        diffusers_unet_map[f"up_blocks.{x}.attentions.{i}.transformer_blocks.{t}.{b}"] = f"output_blocks.{n}.1.transformer_blocks.{t}.{b}"
            if i == num_res_blocks[x]:
                for k in ["weight", "bias"]:
                    diffusers_unet_map[f"up_blocks.{x}.upsamplers.0.conv.{k}"] = f"output_blocks.{n}.{c}.conv.{k}"
            n += 1
    for k in UNET_MAP_BASIC:
        diffusers_unet_map[k[1]] = k[0]
    return diffusers_unet_map


def apply_control1(h: torch.Tensor, control: any, name: str) -> torch.Tensor:
    return h


class UNetModel1(nn.Module):
    def __init__(self, image_size: int, in_channels: int, model_channels: int, out_channels: int, num_res_blocks: list,
                 dropout: float = 0, channel_mult: tuple = (1, 2, 4, 8), conv_resample: bool = True, dims: int = 2,
                 num_classes: int = None, use_checkpoint: bool = False, dtype: torch.dtype = torch.float32,
                 num_heads: int = -1, num_head_channels: int = -1, num_heads_upsample: int = -1,
                 use_scale_shift_norm: bool = False, resblock_updown: bool = False, use_new_attention_order: bool = False,
                 use_spatial_transformer: bool = False, transformer_depth: int = 1, context_dim: int = None,
                 n_embed: int = None, legacy: bool = True, disable_self_attentions: list = None,
                 num_attention_blocks: list = None, disable_middle_self_attn: bool = False,
                 use_linear_in_transformer: bool = False, adm_in_channels: int = None,
                 transformer_depth_middle: int = None, transformer_depth_output: list = None,
                 use_temporal_resblock: bool = False, use_temporal_attention: bool = False,
                 time_context_dim: int = None, extra_ff_mix_layer: bool = False, use_spatial_context: bool = False,
                 merge_strategy: any = None, merge_factor: float = 0.0, video_kernel_size: int = None,
                 disable_temporal_crossattention: bool = False, max_ddpm_temb_period: int = 10000,
                 device: torch.device = None, operations: any = oai_ops):
        super().__init__()
        if context_dim is not None:
            self.context_dim = context_dim
        if num_heads_upsample == -1:
            num_heads_upsample = num_heads
        if num_head_channels == -1:
            assert num_heads != -1

        self.in_channels, self.model_channels, self.out_channels = in_channels, model_channels, out_channels
        self.num_res_blocks, self.dropout, self.channel_mult = num_res_blocks, dropout, channel_mult
        self.conv_resample, self.num_classes, self.use_checkpoint = conv_resample, num_classes, use_checkpoint
        self.dtype, self.num_heads, self.num_head_channels = dtype, num_heads, num_head_channels
        self.num_heads_upsample, self.use_temporal_resblocks = num_heads_upsample, use_temporal_resblock
        self.predict_codebook_ids, self.default_num_video_frames = n_embed is not None, None
        transformer_depth, transformer_depth_output = transformer_depth[:], transformer_depth_output[:]
        time_embed_dim = model_channels * 4

        self.time_embed = nn.Sequential(
            operations.Linear(model_channels, time_embed_dim, dtype=self.dtype, device=device), nn.SiLU(),
            operations.Linear(time_embed_dim, time_embed_dim, dtype=self.dtype, device=device))
        self.input_blocks = nn.ModuleList([sampling.TimestepEmbedSequential1(
            operations.conv_nd(dims, in_channels, model_channels, 3, padding=1, dtype=self.dtype, device=device))])
        self._feature_size, input_block_chans, ch, ds = model_channels, [model_channels], model_channels, 1
        self.double_blocks = nn.ModuleList()

        def make_attn(ch, depth, context_dim, disable_self_attn=False):
            dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
            heads = num_heads if num_head_channels == -1 else ch // num_head_channels
            return transformer.SpatialTransformer(ch, heads, dim_head, depth=depth, context_dim=context_dim,
                disable_self_attn=disable_self_attn, use_linear=use_linear_in_transformer,
                use_checkpoint=use_checkpoint, dtype=self.dtype, device=device, operations=operations)

        def make_res(ch, out_ch=None, down=False, up=False):
            return ResBlock.ResBlock1(channels=ch, emb_channels=time_embed_dim, dropout=dropout,
                out_channels=out_ch, use_checkpoint=use_checkpoint, dims=dims,
                use_scale_shift_norm=use_scale_shift_norm, down=down, up=up,
                dtype=self.dtype, device=device, operations=operations)

        for level, mult in enumerate(channel_mult):
            for nr in range(self.num_res_blocks[level]):
                layers = [make_res(ch, mult * model_channels)]
                ch = mult * model_channels
                num_trans = transformer_depth.pop(0)
                if num_trans > 0 and (not util.exists(num_attention_blocks) or nr < num_attention_blocks[level]):
                    layers.append(make_attn(ch, num_trans, context_dim))
                self.input_blocks.append(sampling.TimestepEmbedSequential1(*layers))
                self._feature_size += ch
                input_block_chans.append(ch)
            if level != len(channel_mult) - 1:
                out_ch = ch
                self.input_blocks.append(sampling.TimestepEmbedSequential1(
                    make_res(ch, out_ch, down=True) if resblock_updown else
                    ResBlock.Downsample1(ch, conv_resample, dims=dims, out_channels=out_ch, dtype=self.dtype, device=device, operations=operations)))
                ch, input_block_chans, ds = out_ch, input_block_chans + [out_ch], ds * 2
                self._feature_size += ch

        dim_head = ch // num_heads if num_head_channels == -1 else num_head_channels
        mid_block = [make_res(ch)]
        self.middle_block = None
        if transformer_depth_middle >= -1:
            if transformer_depth_middle >= 0:
                mid_block += [make_attn(ch, transformer_depth_middle, context_dim, disable_middle_self_attn), make_res(ch)]
            self.middle_block = sampling.TimestepEmbedSequential1(*mid_block)
        self._feature_size += ch

        self.output_blocks = nn.ModuleList([])
        for level, mult in list(enumerate(channel_mult))[::-1]:
            for i in range(self.num_res_blocks[level] + 1):
                ich = input_block_chans.pop()
                layers = [make_res(ch + ich, model_channels * mult)]
                ch = model_channels * mult
                num_trans = transformer_depth_output.pop()
                if num_trans > 0 and (not util.exists(num_attention_blocks) or i < num_attention_blocks[level]):
                    layers.append(make_attn(ch, num_trans, context_dim))
                if level and i == self.num_res_blocks[level]:
                    layers.append(make_res(ch, ch, up=True) if resblock_updown else
                        ResBlock.Upsample1(ch, conv_resample, dims=dims, out_channels=ch, dtype=self.dtype, device=device, operations=operations))
                    ds //= 2
                self.output_blocks.append(sampling.TimestepEmbedSequential1(*layers))
                self._feature_size += ch

        self.out = nn.Sequential(operations.GroupNorm(32, ch, dtype=self.dtype, device=device), nn.SiLU(),
            util.zero_module(operations.conv_nd(dims, model_channels, out_channels, 3, padding=1, dtype=self.dtype, device=device)))

    def forward(self, x: torch.Tensor, timesteps: Optional[torch.Tensor] = None, context: Optional[torch.Tensor] = None,
                y: Optional[torch.Tensor] = None, control: Optional[torch.Tensor] = None,
                transformer_options: Dict[str, Any] = {}, **kwargs: Any) -> torch.Tensor:
        transformer_options["original_shape"], transformer_options["transformer_index"] = list(x.shape), 0
        num_video_frames = kwargs.get("num_video_frames", self.default_num_video_frames)
        image_only_indicator, time_context = kwargs.get("image_only_indicator"), kwargs.get("time_context")
        if self.num_classes is None:
            y = None
        elif y is None:
            raise ValueError("y is required for models with num_classes")
        emb = self.time_embed(sampling_util.timestep_embedding(timesteps, self.model_channels).to(device=x.device, dtype=x.dtype))
        hs, h = [], x
        for id, module in enumerate(self.input_blocks):
            transformer_options["block"] = ("input", id)
            h = ResBlock.forward_timestep_embed1(module, h, emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
            h = apply_control1(h, control, "input")
            hs.append(h)
        transformer_options["block"] = ("middle", 0)
        if self.middle_block is not None:
            h = ResBlock.forward_timestep_embed1(self.middle_block, h, emb, context, transformer_options,
                time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        h = apply_control1(h, control, "middle")
        for id, module in enumerate(self.output_blocks):
            transformer_options["block"] = ("output", id)
            hsp = apply_control1(hs.pop(), control, "output")
            h = torch.cat([h, hsp], dim=1)
            del hsp
            h = ResBlock.forward_timestep_embed1(module, h, emb, context, transformer_options,
                hs[-1].shape if hs else None, time_context=time_context, num_video_frames=num_video_frames, image_only_indicator=image_only_indicator)
        return self.out(h.type(x.dtype))


def detect_unet_config(state_dict: Dict[str, torch.Tensor], key_prefix: str) -> Dict[str, Any]:
    state_dict_keys = list(state_dict.keys())
    
    # MMDIT model
    if f"{key_prefix}joint_blocks.0.context_block.attn.qkv.weight" in state_dict_keys:
        cfg = {"in_channels": state_dict[f"{key_prefix}x_embedder.proj.weight"].shape[1],
               "patch_size": state_dict[f"{key_prefix}x_embedder.proj.weight"].shape[2],
               "depth": state_dict[f"{key_prefix}x_embedder.proj.weight"].shape[0] // 64, "input_size": None}
        if f"{key_prefix}final_layer.linear.weight" in state_dict:
            cfg["out_channels"] = state_dict[f"{key_prefix}final_layer.linear.weight"].shape[0] // (cfg["patch_size"] ** 2)
        if f"{key_prefix}y_embedder.mlp.0.weight" in state_dict_keys:
            cfg["adm_in_channels"] = state_dict[f"{key_prefix}y_embedder.mlp.0.weight"].shape[1]
        if f"{key_prefix}context_embedder.weight" in state_dict_keys:
            w = state_dict[f"{key_prefix}context_embedder.weight"]
            cfg["context_embedder_config"] = {"target": "torch.nn.Linear", "params": {"in_features": w.shape[1], "out_features": w.shape[0]}}
        if f"{key_prefix}pos_embed" in state_dict_keys:
            cfg["num_patches"] = state_dict[f"{key_prefix}pos_embed"].shape[1]
            cfg["pos_embed_max_size"] = round(math.sqrt(cfg["num_patches"]))
        if f"{key_prefix}joint_blocks.0.context_block.attn.ln_q.weight" in state_dict_keys:
            cfg["qk_norm"] = "rms"
        cfg["pos_embed_scaling_factor"] = None
        if f"{key_prefix}context_processor.layers.0.attn.qkv.weight" in state_dict_keys:
            cfg["context_processor_layers"] = transformer.count_blocks(state_dict_keys, f"{key_prefix}context_processor.layers." + "{}.")
        return cfg

    # Stable Cascade
    if f"{key_prefix}clf.1.weight" in state_dict_keys:
        cfg = {}
        if f"{key_prefix}clip_txt_mapper.weight" in state_dict_keys:
            cfg["stable_cascade_stage"] = "c"
            w = state_dict[f"{key_prefix}clip_txt_mapper.weight"]
            if w.shape[0] == 1536:
                cfg.update({"c_cond": 1536, "c_hidden": [1536, 1536], "nhead": [24, 24], "blocks": [[4, 12], [12, 4]]})
            elif w.shape[0] == 2048:
                cfg["c_cond"] = 2048
        elif f"{key_prefix}clip_mapper.weight" in state_dict_keys:
            cfg["stable_cascade_stage"] = "b"
            w = state_dict[f"{key_prefix}down_blocks.1.0.channelwise.0.weight"]
            if w.shape[-1] == 640:
                cfg.update({"c_hidden": [320, 640, 1280, 1280], "nhead": [-1, -1, 20, 20], "blocks": [[2, 6, 28, 6], [6, 28, 6, 2]], "block_repeat": [[1, 1, 1, 1], [3, 3, 2, 2]]})
            elif w.shape[-1] == 576:
                cfg.update({"c_hidden": [320, 576, 1152, 1152], "nhead": [-1, 9, 18, 18], "blocks": [[2, 4, 14, 4], [4, 14, 4, 2]], "block_repeat": [[1, 1, 1, 1], [2, 2, 2, 2]]})
        return cfg

    # Stable Audio DIT
    if f"{key_prefix}transformer.rotary_pos_emb.inv_freq" in state_dict_keys:
        return {"audio_model": "dit1.0"}

    # Aura Flow DIT
    if f"{key_prefix}double_layers.0.attn.w1q.weight" in state_dict_keys:
        double_layers = transformer.count_blocks(state_dict_keys, f"{key_prefix}double_layers." + "{}.")
        single_layers = transformer.count_blocks(state_dict_keys, f"{key_prefix}single_layers." + "{}.")
        return {"max_seq": state_dict[f"{key_prefix}positional_encoding"].shape[1],
                "cond_seq_dim": state_dict[f"{key_prefix}cond_seq_linear.weight"].shape[1],
                "n_double_layers": double_layers, "n_layers": double_layers + single_layers}

    # Hunyuan DiT
    if f"{key_prefix}mlp_t5.0.weight" in state_dict_keys:
        cfg = {"image_model": "hydit", "depth": transformer.count_blocks(state_dict_keys, f"{key_prefix}blocks." + "{}."),
               "hidden_size": state_dict[f"{key_prefix}x_embedder.proj.weight"].shape[0]}
        if cfg["hidden_size"] == 1408 and cfg["depth"] == 40:
            cfg["mlp_ratio"] = 4.3637
        if state_dict[f"{key_prefix}extra_embedder.0.weight"].shape[1] == 3968:
            cfg.update({"size_cond": True, "use_style_cond": True, "image_model": "hydit1"})
        return cfg

    # Flux
    if f"{key_prefix}double_blocks.0.img_attn.norm.key_norm.scale" in state_dict_keys:
        return {"image_model": "flux", "in_channels": 16, "vec_in_dim": 768, "context_in_dim": 4096,
                "hidden_size": 3072, "mlp_ratio": 4.0, "num_heads": 24,
                "depth": transformer.count_blocks(state_dict_keys, f"{key_prefix}double_blocks." + "{}."),
                "depth_single_blocks": transformer.count_blocks(state_dict_keys, f"{key_prefix}single_blocks." + "{}."),
                "axes_dim": [16, 56, 56], "theta": 10000, "qkv_bias": True,
                "guidance_embed": f"{key_prefix}guidance_in.in_layer.weight" in state_dict_keys}

    if f"{key_prefix}input_blocks.0.0.weight" not in state_dict_keys:
        return None

    # Standard UNet
    cfg = {"use_checkpoint": False, "image_size": 32, "use_spatial_transformer": True, "legacy": False}
    if f"{key_prefix}label_emb.0.0.weight" in state_dict_keys:
        cfg["num_classes"], cfg["adm_in_channels"] = "sequential", state_dict[f"{key_prefix}label_emb.0.0.weight"].shape[1]
    else:
        cfg["adm_in_channels"] = None

    model_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[0]
    in_channels = state_dict[f"{key_prefix}input_blocks.0.0.weight"].shape[1]
    out_channels = state_dict.get(f"{key_prefix}out.2.weight", torch.zeros(4)).shape[0] or 4

    num_res_blocks, channel_mult, transformer_depth, transformer_depth_output = [], [], [], []
    context_dim, use_linear_in_transformer = None, False
    current_res, last_res_blocks, last_channel_mult = 1, 0, 0
    input_block_count = transformer.count_blocks(state_dict_keys, f"{key_prefix}input_blocks" + ".{}.")

    for count in range(input_block_count):
        prefix = f"{key_prefix}input_blocks.{count}."
        prefix_output = f"{key_prefix}output_blocks.{input_block_count - count - 1}."
        block_keys = sorted([k for k in state_dict_keys if k.startswith(prefix)])
        if not block_keys:
            break
        block_keys_output = sorted([k for k in state_dict_keys if k.startswith(prefix_output)])

        if f"{prefix}0.op.weight" in block_keys:
            num_res_blocks.append(last_res_blocks)
            channel_mult.append(last_channel_mult)
            current_res *= 2
            last_res_blocks, last_channel_mult = 0, 0
            out = transformer.calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
            transformer_depth_output.append(out[0] if out else 0)
        else:
            if f"{prefix}0.in_layers.0.weight" in block_keys:
                last_res_blocks += 1
                last_channel_mult = state_dict[f"{prefix}0.out_layers.3.weight"].shape[0] // model_channels
                out = transformer.calculate_transformer_depth(prefix, state_dict_keys, state_dict)
                if out:
                    transformer_depth.append(out[0])
                    if context_dim is None:
                        context_dim, use_linear_in_transformer = out[1], out[2]
                else:
                    transformer_depth.append(0)
            if f"{prefix_output}0.in_layers.0.weight" in block_keys_output:
                out = transformer.calculate_transformer_depth(prefix_output, state_dict_keys, state_dict)
                transformer_depth_output.append(out[0] if out else 0)

    num_res_blocks.append(last_res_blocks)
    channel_mult.append(last_channel_mult)
    if f"{key_prefix}middle_block.1.proj_in.weight" in state_dict_keys:
        transformer_depth_middle = transformer.count_blocks(state_dict_keys, f"{key_prefix}middle_block.1.transformer_blocks." + "{}")
    elif f"{key_prefix}middle_block.0.in_layers.0.weight" in state_dict_keys:
        transformer_depth_middle = -1
    else:
        transformer_depth_middle = -2

    cfg.update({"in_channels": in_channels, "out_channels": out_channels, "model_channels": model_channels,
                "num_res_blocks": num_res_blocks, "transformer_depth": transformer_depth,
                "transformer_depth_output": transformer_depth_output, "channel_mult": channel_mult,
                "transformer_depth_middle": transformer_depth_middle, "use_linear_in_transformer": use_linear_in_transformer,
                "context_dim": context_dim, "use_temporal_resblock": False, "use_temporal_attention": False})
    return cfg


def model_config_from_unet_config(unet_config: Dict[str, Any], state_dict: Optional[Dict[str, torch.Tensor]] = None) -> Any:
    from src.SD15 import SD15
    for model_config in SD15.models:
        if model_config.matches(unet_config, state_dict):
            return model_config(unet_config)
    logging.error(f"no match {unet_config}")
    return None


def model_config_from_unet(state_dict: Dict[str, torch.Tensor], unet_key_prefix: str, use_base_if_no_match: bool = False) -> Any:
    unet_config = detect_unet_config(state_dict, unet_key_prefix)
    return model_config_from_unet_config(unet_config, state_dict) if unet_config else None


def unet_dtype1(device: Optional[torch.device] = None, model_params: int = 0,
                supported_dtypes: List[torch.dtype] = [torch.float16, torch.bfloat16, torch.float32]) -> torch.dtype:
    return torch.float16
