import torch
from src.Utilities import util
from src.Device import Device
from src.cond import cond_util
from src.sample import ksampler_util


class CONDRegular:
    """#### Class representing a regular condition."""

    def __init__(self, cond: torch.Tensor):
        """#### Initialize the CONDRegular class.

        #### Args:
            - `cond` (torch.Tensor): The condition tensor.
        """
        self.cond = cond

    def _copy_with(self, cond: torch.Tensor) -> "CONDRegular":
        """#### Copy the condition with a new condition.

        #### Args:
            - `cond` (torch.Tensor): The new condition.

        #### Returns:
            - `CONDRegular`: The copied condition.
        """
        return self.__class__(cond)

    def process_cond(
        self, batch_size: int, device: torch.device, **kwargs
    ) -> "CONDRegular":
        """#### Process the condition.

        #### Args:
            - `batch_size` (int): The batch size.
            - `device` (torch.device): The device.

        #### Returns:
            - `CONDRegular`: The processed condition.
        """
        return self._copy_with(
            util.repeat_to_batch_size(self.cond, batch_size).to(device)
        )

    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.

        #### Args:
            - `other` (CONDRegular): The other condition.

        #### Returns:
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        if self.cond.shape != other.cond.shape:
            return False
        return True

    def concat(self, others: list) -> torch.Tensor:
        """#### Concatenate conditions.

        #### Args:
            - `others` (list): The list of other conditions.

        #### Returns:
            - `torch.Tensor`: The concatenated conditions.
        """
        conds = [self.cond]
        for x in others:
            conds.append(x.cond)
        return torch.cat(conds)


class CONDCrossAttn(CONDRegular):
    """#### Class representing a cross-attention condition."""

    def can_concat(self, other: "CONDRegular") -> bool:
        """#### Check if conditions can be concatenated.

        #### Args:
            - `other` (CONDRegular): The other condition.

        #### Returns:
            - `bool`: True if conditions can be concatenated, False otherwise.
        """
        s1 = self.cond.shape
        s2 = other.cond.shape
        if s1 != s2:
            if s1[0] != s2[0] or s1[2] != s2[2]:  # these 2 cases should not happen
                return False

            mult_min = torch.lcm(s1[1], s2[1])
            diff = mult_min // min(s1[1], s2[1])
            if (
                diff > 4
            ):  # arbitrary limit on the padding because it's probably going to impact performance negatively if it's too much
                return False
        return True

    def concat(self, others: list) -> torch.Tensor:
        """Optimized version of cross-attention condition concatenation."""
        conds = [self.cond]
        shapes = [self.cond.shape[1]]

        # Collect all conditions and their shapes
        for x in others:
            conds.append(x.cond)
            shapes.append(x.cond.shape[1])

        # Calculate LCM more efficiently
        crossattn_max_len = util.lcm_of_list(shapes)

        # Process and concat in one step where possible
        if all(c.shape[1] == shapes[0] for c in conds):
            # All same length, simple concatenation
            return torch.cat(conds)
        else:
            # Process conditions that need repeating
            out = []
            for c in conds:
                if c.shape[1] < crossattn_max_len:
                    repeat_factor = crossattn_max_len // c.shape[1]
                    # Use repeat instead of individual operations
                    c = c.repeat(1, repeat_factor, 1)
                out.append(c)
            return torch.cat(out)


def convert_cond(cond: list) -> list:
    """#### Convert conditions to cross-attention conditions.

    #### Args:
        - `cond` (list): The list of conditions.

    #### Returns:
        - `list`: The converted conditions.
    """
    out = []
    for c in cond:
        temp = c[1].copy()
        model_conds = temp.get("model_conds", {})
        if c[0] is not None:
            model_conds["c_crossattn"] = CONDCrossAttn(c[0])
            temp["cross_attn"] = c[0]
        temp["model_conds"] = model_conds
        out.append(temp)
    return out


def calc_cond_batch(
    model: object,
    conds: list,
    x_in: torch.Tensor,
    timestep: torch.Tensor,
    model_options: dict,
) -> list:
    """#### Calculate the condition batch.

    #### Args:
        - `model` (object): The model.
        - `conds` (list): The list of conditions.
        - `x_in` (torch.Tensor): The input tensor.
        - `timestep` (torch.Tensor): The timestep tensor.
        - `model_options` (dict): The model options.

    #### Returns:
        - `list`: The calculated condition batch.
    """
    out_conds = []
    out_counts = []
    to_run = []

    for i in range(len(conds)):
        out_conds.append(torch.zeros_like(x_in))
        out_counts.append(torch.ones_like(x_in) * 1e-37)

        cond = conds[i]
        if cond is not None:
            for x in cond:
                p = ksampler_util.get_area_and_mult(x, x_in, timestep)
                if p is None:
                    continue

                to_run += [(p, i)]

    while len(to_run) > 0:
        first = to_run[0]
        first_shape = first[0][0].shape
        to_batch_temp = []
        for x in range(len(to_run)):
            if cond_util.can_concat_cond(to_run[x][0], first[0]):
                to_batch_temp += [x]

        to_batch_temp.reverse()
        to_batch = to_batch_temp[:1]

        free_memory = Device.get_free_memory(x_in.device)
        for i in range(1, len(to_batch_temp) + 1):
            batch_amount = to_batch_temp[: len(to_batch_temp) // i]
            input_shape = [len(batch_amount) * first_shape[0]] + list(first_shape)[1:]
            if model.memory_required(input_shape) * 1.5 < free_memory:
                to_batch = batch_amount
                break

        input_x = []
        mult = []
        c = []
        cond_or_uncond = []
        area = []
        control = None
        patches = None
        for x in to_batch:
            o = to_run.pop(x)
            p = o[0]
            input_x.append(p.input_x)
            mult.append(p.mult)
            c.append(p.conditioning)
            area.append(p.area)
            cond_or_uncond.append(o[1])
            control = p.control
            patches = p.patches

        batch_chunks = len(cond_or_uncond)
        input_x = torch.cat(input_x)
        c = cond_util.cond_cat(c)
        timestep_ = torch.cat([timestep] * batch_chunks)

        if control is not None:
            c["control"] = control.get_control(
                input_x, timestep_, c, len(cond_or_uncond)
            )

        transformer_options = {}
        if "transformer_options" in model_options:
            transformer_options = model_options["transformer_options"].copy()

        if patches is not None:
            if "patches" in transformer_options:
                cur_patches = transformer_options["patches"].copy()
                for p in patches:
                    if p in cur_patches:
                        cur_patches[p] = cur_patches[p] + patches[p]
                    else:
                        cur_patches[p] = patches[p]
                transformer_options["patches"] = cur_patches
            else:
                transformer_options["patches"] = patches

        transformer_options["cond_or_uncond"] = cond_or_uncond[:]
        transformer_options["sigmas"] = timestep

        c["transformer_options"] = transformer_options

        if "model_function_wrapper" in model_options:
            output = model_options["model_function_wrapper"](
                model.apply_model,
                {
                    "input": input_x,
                    "timestep": timestep_,
                    "c": c,
                    "cond_or_uncond": cond_or_uncond,
                },
            ).chunk(batch_chunks)
        else:
            output = model.apply_model(input_x, timestep_, **c).chunk(batch_chunks)

        for o in range(batch_chunks):
            cond_index = cond_or_uncond[o]
            a = area[o]
            if a is None:
                out_conds[cond_index] += output[o] * mult[o]
                out_counts[cond_index] += mult[o]
            else:
                out_c = out_conds[cond_index]
                out_cts = out_counts[cond_index]
                dims = len(a) // 2
                for i in range(dims):
                    out_c = out_c.narrow(i + 2, a[i + dims], a[i])
                    out_cts = out_cts.narrow(i + 2, a[i + dims], a[i])
                out_c += output[o] * mult[o]
                out_cts += mult[o]

    # Vectorize the division at the end
    for i in range(len(out_conds)):
        # Inplace division is already efficient
        out_conds[i].div_(out_counts[i])  # Using .div_ instead of /= for clarity

    return out_conds


def encode_model_conds(
    model_function: callable,
    conds: list,
    noise: torch.Tensor,
    device: torch.device,
    prompt_type: str,
    **kwargs,
) -> list:
    """#### Encode model conditions.

    #### Args:
        - `model_function` (callable): The model function.
        - `conds` (list): The list of conditions.
        - `noise` (torch.Tensor): The noise tensor.
        - `device` (torch.device): The device.
        - `prompt_type` (str): The prompt type.
        - `**kwargs`: Additional keyword arguments.

    #### Returns:
        - `list`: The encoded model conditions.
    """
    for t in range(len(conds)):
        x = conds[t]
        params = x.copy()
        params["device"] = device
        params["noise"] = noise
        default_width = None
        if len(noise.shape) >= 4:  # TODO: 8 multiple should be set by the model
            default_width = noise.shape[3] * 8
        params["width"] = params.get("width", default_width)
        params["height"] = params.get("height", noise.shape[2] * 8)
        params["prompt_type"] = params.get("prompt_type", prompt_type)
        for k in kwargs:
            if k not in params:
                params[k] = kwargs[k]

        out = model_function(**params)
        x = x.copy()
        model_conds = x["model_conds"].copy()
        for k in out:
            model_conds[k] = out[k]
        x["model_conds"] = model_conds
        conds[t] = x
    return conds


def resolve_areas_and_cond_masks_multidim(conditions, dims, device):
    """Optimized version that processes areas and masks more efficiently"""
    for i in range(len(conditions)):
        c = conditions[i]
        # Process area
        if "area" in c:
            area = c["area"]
            if area[0] == "percentage":
                # Vectorized calculation of area dimensions
                a = area[1:]
                a_len = len(a) // 2

                # Calculate all dimensions at once using tensor operations
                dims_tensor = torch.tensor(dims, device="cpu")
                first_part = torch.tensor(a[:a_len], device="cpu") * dims_tensor
                second_part = torch.tensor(a[a_len:], device="cpu") * dims_tensor

                # Convert to rounded integers and tuple
                first_part = torch.max(
                    torch.ones_like(first_part), torch.round(first_part)
                )
                second_part = torch.round(second_part)

                # Create the new area tuple
                new_area = tuple(first_part.int().tolist()) + tuple(
                    second_part.int().tolist()
                )

                # Create a modified copy with the new area
                modified = c.copy()
                modified["area"] = new_area
                conditions[i] = modified

        # Process mask
        if "mask" in c:
            modified = c.copy()
            mask = c["mask"].to(device=device)

            # Combine dimension checks and unsqueeze operation
            if len(mask.shape) == len(dims):
                mask = mask.unsqueeze(0)

            # Only interpolate if needed
            if mask.shape[1:] != dims:
                # Optimize interpolation by ensuring mask is in the right format for the operation
                if len(mask.shape) == 3 and mask.shape[0] == 1:
                    # Already in the right format for interpolation
                    mask = torch.nn.functional.interpolate(
                        mask.unsqueeze(1),
                        size=dims,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)
                else:
                    # Ensure mask is properly formatted for interpolation
                    mask = torch.nn.functional.interpolate(
                        mask
                        if len(mask.shape) > 3 and mask.shape[1] == 1
                        else mask.unsqueeze(1),
                        size=dims,
                        mode="bilinear",
                        align_corners=False,
                    ).squeeze(1)

            modified["mask"] = mask
            conditions[i] = modified


def process_conds(
    model: object,
    noise: torch.Tensor,
    conds: dict,
    device: torch.device,
    latent_image: torch.Tensor = None,
    denoise_mask: torch.Tensor = None,
    seed: int = None,
) -> dict:
    """#### Process conditions.

    #### Args:
        - `model` (object): The model.
        - `noise` (torch.Tensor): The noise tensor.
        - `conds` (dict): The conditions.
        - `device` (torch.device): The device.
        - `latent_image` (torch.Tensor, optional): The latent image tensor. Defaults to None.
        - `denoise_mask` (torch.Tensor, optional): The denoise mask tensor. Defaults to None.
        - `seed` (int, optional): The seed. Defaults to None.

    #### Returns:
        - `dict`: The processed conditions.
    """
    for k in conds:
        conds[k] = conds[k][:]
        resolve_areas_and_cond_masks_multidim(conds[k], noise.shape[2:], device)

    for k in conds:
        ksampler_util.calculate_start_end_timesteps(model, conds[k])

    if hasattr(model, "extra_conds"):
        for k in conds:
            conds[k] = encode_model_conds(
                model.extra_conds,
                conds[k],
                noise,
                device,
                k,
                latent_image=latent_image,
                denoise_mask=denoise_mask,
                seed=seed,
            )

    # make sure each cond area has an opposite one with the same area
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
                    list(
                        filter(
                            lambda c: c.get("control_apply_to_uncond", False) is True,
                            positive,
                        )
                    ),
                    conds[k],
                    "control",
                    lambda cond_cnets, x: cond_cnets[x],
                )
                ksampler_util.apply_empty_x_to_equal_area(
                    positive, conds[k], "gligen", lambda cond_cnets, x: cond_cnets[x]
                )

    return conds
