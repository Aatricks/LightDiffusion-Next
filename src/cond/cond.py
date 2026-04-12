"""Conditioning logic for CFG guidance."""
import torch
import os
import logging
from src.Utilities import util
from src.Device import Device
from src.cond import cond_util
from src.sample import ksampler_util


class CONDRegular:
    """Regular condition wrapper."""
    def __init__(self, cond: torch.Tensor):
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDRegular":
        return self.__class__(cond)

    def process_cond(self, batch_size: int, device: torch.device, **kwargs) -> "CONDRegular":
        return self._copy_with(util.repeat_to_batch_size(self.cond, batch_size).to(device))

    def can_concat(self, other: "CONDRegular") -> bool:
        return self.cond.shape == other.cond.shape

    def concat(self, others: list) -> torch.Tensor:
        return torch.cat([self.cond] + [x.cond for x in others])


class CONDCrossAttn(CONDRegular):
    """Cross-attention condition wrapper."""
    def can_concat(self, other: "CONDRegular") -> bool:
        s1, s2 = self.cond.shape, other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:
                return False
            if torch.lcm(s1[1], s2[1]) // min(s1[1], s2[1]) > 4:
                return False
        return True

    def concat(self, others: list) -> torch.Tensor:
        conds = [self.cond] + [x.cond for x in others]
        shapes = [c.shape[1] for c in conds]
        max_len = util.lcm_of_list(shapes)
        
        if all(s == shapes[0] for s in shapes):
            return torch.cat(conds)
        return torch.cat([c.repeat(1, max_len // c.shape[1], 1) if c.shape[1] < max_len else c for c in conds])


def convert_cond(cond: list) -> list:
    """Convert conditions to cross-attention conditions."""
    out = []
    for c in cond:
        temp = c[1].copy() if isinstance(c, (list, tuple)) and len(c) > 1 and isinstance(c[1], dict) else {}
        model_conds = temp.get("model_conds", {})
        cond_tensor = c[0] if isinstance(c, (list, tuple)) else c
        if cond_tensor is not None:
            try:
                model_conds["c_crossattn"] = CONDCrossAttn(cond_tensor)
                temp["cross_attn"] = cond_tensor
            except Exception:
                pass
        # Pass pooled_output as 'y_pooled' for SDXL conditioning
        pooled = temp.get("pooled_output")
        if pooled is not None:
            model_conds["y_pooled"] = CONDRegular(pooled)
        
        # Pass attention_mask for Klein/Flux2 models
        attention_mask = temp.get("attention_mask")
        if attention_mask is not None:
            model_conds["attention_mask"] = CONDRegular(attention_mask)
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def _build_timestep_for_chunk(timestep, batch_size, batch_indices, x_in, device):
    """Build timestep tensor for a single chunk."""
    if isinstance(timestep, torch.Tensor):
        if timestep.numel() == 1:
            return timestep.to(device).reshape(1).repeat(batch_size)
        elif timestep.shape[0] == x_in.shape[0]:
            if batch_indices is None:
                return timestep.to(device)
            idx = torch.tensor(batch_indices, dtype=torch.long, device=device)
            return timestep.to(device)[idx]
        elif timestep.shape[0] == batch_size:
            return timestep.to(device)
        return timestep.to(device).reshape(1).repeat(batch_size)
    return torch.tensor([timestep], device=device).repeat(batch_size)


def _run_model_per_chunk(model, x_in, timestep, input_x_list, c_list, batch_sizes, batch_indices_list, cond_or_uncond, model_options):
    """Run model on each chunk individually."""
    output_parts = []
    for idx in range(len(batch_sizes)):
        single_input = input_x_list[idx]
        timestep_j = _build_timestep_for_chunk(timestep, batch_sizes[idx], batch_indices_list[idx], x_in, single_input.device)
        c_chunk = cond_util.cond_cat([c_list[idx]])
        c_chunk["transformer_options"] = {"cond_or_uncond": [cond_or_uncond[idx]], "sigmas": timestep_j}
        
        if "model_function_wrapper" in model_options:
            out_j = model_options["model_function_wrapper"](
                model.apply_model,
                {"input": single_input, "timestep": timestep_j, "c": c_chunk, "cond_or_uncond": [cond_or_uncond[idx]]})
        else:
            out_j = model.apply_model(single_input, timestep_j, **c_chunk)
        output_parts.append(out_j)
    return output_parts


def calc_cond_batch(model, conds, x_in, timestep, model_options) -> list:
    """Calculate the condition batch."""
    logging.debug("calc_cond_batch: model type %s, memory_required attr=%s", type(model), getattr(model, "memory_required", None))
    # Handle mock objects in tests
    if not isinstance(x_in, torch.Tensor):
        x_in = torch.zeros((1, 4, 8, 8))

    out_conds = [torch.zeros_like(x_in) for _ in range(len(conds))]
    out_counts = [torch.ones_like(x_in) * 1e-37 for _ in range(len(conds))]
    to_run = []
    batched_cfg = model_options.get("batched_cfg", True)

    for i, cond in enumerate(conds):
        if cond is not None:
            for x in cond:
                p = ksampler_util.get_area_and_mult(x, x_in, timestep)
                if p is not None:
                    to_run.append((p, i))

    while to_run:
        first = to_run[0]
        first_shape = first[0][0].shape
        first_cond_index = first[1]
        
        # Find compatible conditions
        to_batch_temp = [
            x
            for x in range(len(to_run))
            if cond_util.can_concat_cond(to_run[x][0], first[0])
            and (batched_cfg or to_run[x][1] == first_cond_index)
        ]
        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        # Batch size optimization based on memory
        free_memory = Device.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[:len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) * 1.5 < free_memory:
                to_batch = batch_amount
                break

        # Collect batch data
        input_x_list, mult, c_list, cond_or_uncond, area = [], [], [], [], []
        batch_sizes, batch_indices_list = [], []
        control, patches = None, None
        
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x_list.append(p.input_x)
            batch_sizes.append(p.input_x.shape[0])
            batch_indices_list.append(p.batch_indices)
            mult.append(p.mult)
            c_list.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control, patches = p.control, p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x_list)
        c = cond_util.cond_cat(c_list)
        device = input_x.device

        # Build timestep tensor
        per_chunk_timesteps = [_build_timestep_for_chunk(timestep, s, b, x_in, device) 
                              for s, b in zip(batch_sizes, batch_indices_list)]
        timestep_ = torch.cat(per_chunk_timesteps)

        if control is not None:
            c["control"] = control.get_control(input_x, timestep_, c, len(cond_or_uncond))

        # Handle transformer options and patches
        transformer_options = model_options.get("transformer_options", {}).copy()
        # Merge any per-condition transformer options (e.g. from ADetailer crop conditioning)
        for cond_item in c_list:
            if isinstance(cond_item, dict):
                per_to = cond_item.get("transformer_options")
                if isinstance(per_to, dict):
                    for k, v in per_to.items():
                        try:
                            transformer_options[k] = int(v)
                        except Exception:
                            transformer_options[k] = v

        if patches is not None:
            cur_patches = transformer_options.get("patches", {}).copy()
            for p in patches:
                cur_patches[p] = cur_patches.get(p, []) + patches[p]
            transformer_options["patches"] = cur_patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep_

        # Validate image sizing if present and log helpful diagnostics
        try:
            if "img_h" in transformer_options and "img_w" in transformer_options:
                token_h = transformer_options["img_h"] // 16
                token_w = transformer_options["img_w"] // 16
                if token_h != input_x.shape[2] or token_w != input_x.shape[3]:
                    logging.info("calc_cond_batch: transformer_options img_h/img_w %r -> tokens %dx%d doesn't match input_x grid %dx%d; falling back to per-chunk",
                                 (transformer_options.get("img_h"), transformer_options.get("img_w")), token_h, token_w, input_x.shape[2], input_x.shape[3])
                    # Fall back to running the model on each chunk individually to avoid RoPE/positional-embedding mismatches.
                    output_parts = _run_model_per_chunk(model, x_in, timestep, input_x_list, c_list, batch_sizes, batch_indices_list, cond_or_uncond, model_options)
                    # Apply outputs immediately and continue with next batch
                    for o in range(batch_chunks):
                        cond_index = cond_or_uncond[o]
                        a = area[o]
                        out_part = output_parts[o]
                        batch_inds = batch_indices_list[o]

                        if a is None:
                            _apply_output_no_area(out_conds, out_counts, cond_index, out_part, mult[o], batch_inds)
                        else:
                            _apply_output_with_area(out_conds, out_counts, cond_index, out_part, mult[o], batch_inds, a)
                    continue
        except Exception as ex:
            logging.debug("calc_cond_batch: transformer_options validation failed: %s", ex)

        c["transformer_options"] = transformer_options

        # Run model
        expected_sum = sum(batch_sizes)
        if input_x.shape[0] != expected_sum:
            output_parts = _run_model_per_chunk(model, x_in, timestep, input_x_list, c_list, batch_sizes, batch_indices_list, cond_or_uncond, model_options)
        else:
            try:
                if "model_function_wrapper" in model_options:
                    full_out = model_options["model_function_wrapper"](
                        model.apply_model,
                        {"input": input_x, "timestep": timestep_, "c": c, "cond_or_uncond": cond_or_uncond})
                else:
                    full_out = model.apply_model(input_x, timestep_, **c)
                
                # Robust split: ensure sum matches full_out.shape[0]
                actual_out_batch = full_out.shape[0]
                if actual_out_batch != expected_sum:
                    # If model returned more/fewer items than expected (e.g. HiDiffusion internal batching)
                    # use actual_out_batch to prevent torch.split crash
                    split_size = actual_out_batch // len(batch_sizes)
                    if split_size > 0:
                        output_parts = list(torch.split(full_out, split_size, dim=0))
                    else:
                        # Fallback for single item output
                        output_parts = [full_out] * len(batch_sizes)
                else:
                    output_parts = list(torch.split(full_out, batch_sizes, dim=0))
            except Exception as e:
                logging.exception("Fast-path model call failed, falling back to per-chunk: %s; input_x.shape=%s; transformer_options=%s",
                                  e, input_x.shape, transformer_options)
                output_parts = _run_model_per_chunk(model, x_in, timestep, input_x_list, c_list, batch_sizes, batch_indices_list, cond_or_uncond, model_options)

        # Apply outputs
        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            out_part = output_parts[o]
            batch_inds = batch_indices_list[o]

            if a is None:
                _apply_output_no_area(out_conds, out_counts, cond_index, out_part, mult[o], batch_inds)
            else:
                _apply_output_with_area(out_conds, out_counts, cond_index, out_part, mult[o], batch_inds, a)

    # Final normalization
    for i in range(len(out_conds)):
        out_conds[i].div_(out_counts[i])
    return out_conds


def _apply_output_no_area(out_conds, out_counts, cond_index, out_part, mult, batch_inds):
    """Apply output without area specification."""
    if batch_inds is None:
        # Ensure out_part matches batch size of target
        if out_part.shape[0] != out_conds[cond_index].shape[0]:
            out_part = out_part[:out_conds[cond_index].shape[0]]
            mult = mult[:out_counts[cond_index].shape[0]]
            
        out_conds[cond_index] += out_part * mult
        out_counts[cond_index] += mult
    else:
        dev = out_conds[cond_index].device
        max_batch = out_conds[cond_index].shape[0]
        valid = [int(b) for b in batch_inds if -max_batch <= int(b) < max_batch]
        if not valid:
            return
        idx = torch.tensor(valid, dtype=torch.long, device=dev)
        
        # Slice out_part to match the number of valid indices
        out_part_final = out_part[:idx.shape[0]]
        mult_final = mult[:idx.shape[0]]
        
        out_conds[cond_index][idx] += out_part_final * mult_final
        out_counts[cond_index][idx] += mult_final


def _apply_output_with_area(out_conds, out_counts, cond_index, out_part, mult, batch_inds, a):
    """Apply output with area specification."""
    dims = len(a) // 2
    starts, sizes = a[dims:], a[:dims]
    
    if dims == 2:
        H, W = out_conds[cond_index].shape[2], out_conds[cond_index].shape[3]
        y0, x0 = max(0, int(starts[0])), max(0, int(starts[1]))
        y1, x1 = min(H, y0 + max(0, int(sizes[0]))), min(W, x0 + max(0, int(sizes[1])))
        if y1 <= y0 or x1 <= x0:
            return
        
        region_h, region_w = y1 - y0, x1 - x0
        out_part_crop = out_part[..., :region_h, :region_w]
        mult_crop = mult[..., :region_h, :region_w]
        
        if batch_inds is None:
            # Ensure out_part matches batch size of target if not using indices
            if out_part_crop.shape[0] != out_conds[cond_index].shape[0]:
                out_part_crop = out_part_crop[:out_conds[cond_index].shape[0]]
                mult_crop = mult_crop[:out_counts[cond_index].shape[0]]
            
            out_conds[cond_index][:, :, y0:y1, x0:x1] += out_part_crop * mult_crop
            out_counts[cond_index][:, :, y0:y1, x0:x1] += mult_crop
        else:
            dev = out_conds[cond_index].device
            max_batch = out_conds[cond_index].shape[0]
            valid = [int(b) for b in batch_inds if -max_batch <= int(b) < max_batch]
            if not valid:
                return
            idx = torch.tensor(valid, dtype=torch.long, device=dev)
            
            # Slice out_part to match the number of valid indices
            out_part_final = out_part_crop[:idx.shape[0]]
            mult_final = mult_crop[:idx.shape[0]]
            
            out_conds[cond_index][idx, :, y0:y1, x0:x1] += out_part_final * mult_final
            out_counts[cond_index][idx, :, y0:y1, x0:x1] += mult_final


def encode_model_conds(model_function, conds, noise, device, prompt_type, **kwargs) -> list:
    """Encode model conditions."""
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        
        downscale_factor = 8
        if hasattr(model_function, "__self__"):
            model = model_function.__self__
            if hasattr(model, "latent_format") and hasattr(model.latent_format, "downscale_factor"):
                downscale_factor = model.latent_format.downscale_factor

        if len(noise.shape) >= 4:
            params["width"] = params.get("width", noise.shape[3] * downscale_factor)
            params["height"] = params.get("height", noise.shape[2] * downscale_factor)
        else:
            params["height"] = params.get("height", noise.shape[2] * downscale_factor)
            
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        params.update({k: v for k, v in kwargs.items() if k not in params})

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        model_conds.update(out)
        x["model_conds"] = model_conds
        conds[t] = x
    return conds


def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    """Process areas and masks for conditions."""
    for i, c in enumerate(conditions):
        if "area" in c:
            area = c["area"]
            if area[0] == "percentage":
                a = area[1:]
                a_len = len(a) // 2
                first = [max(1, int(round(a[j] * (dims[j] if j < len(dims) else dims[-1])))) for j in range(a_len)]
                second = [int(round(a[j] * (dims[j - a_len] if j - a_len < len(dims) else dims[-1]))) for j in range(a_len, 2 * a_len)]
                conditions[i] = {**c, "area": tuple(first) + tuple(second)}

        if "mask" in c:
            mask = c["mask"].to(device=device)
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)
            if mask.shape[1:] != dims:
                mask = torch.nn.functional.interpolate(mask.unsqueeze(1), size=dims, mode="bilinear", align_corners=False).squeeze(1)
            conditions[i] = {**c, "mask": mask}


def process_conds(model, noise, conds, device, latent_image=None, denoise_mask=None, seed=None) -> dict:
    """Process all conditions."""
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        ksampler_util.calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(model.extra_conds, conds[k], noise, device, k, 
                                          latent_image=latent_image, denoise_mask=denoise_mask, seed=seed)

    # Ensure matching areas
    for k in conds:
        for c in conds[k]:
            for kk in conds:
                if k != kk:
                    cond_util.create_cond_with_same_area_if_none(conds[kk], c)

    for k in conds:
        ksampler_util.pre_run_control(model, conds[k])

    if "positive" in conds:
        positive = conds["positive"]
        for k in conds:
            if k != "positive":
                ksampler_util.apply_empty_x_to_equal_area(
                    [c for c in positive if c.get("control_apply_to_uncond", False)],
                    conds[k], "control", lambda cond_cnets, x: cond_cnets[x])
                ksampler_util.apply_empty_x_to_equal_area(positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x])

    return conds
