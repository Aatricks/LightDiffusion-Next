
import logging
import torch
from modules import Device, ModelPatcher, VariationalAE, unet, util, Clip

def load_checkpoint_guess_config(
    ckpt_path,
    output_vae=True,
    output_clip=True,
    output_clipvision=False,
    embedding_directory=None,
    output_model=True,
):
    sd = util.load_torch_file(ckpt_path)
    sd_keys = sd.keys()
    clip = None
    clipvision = None
    vae = None
    model = None
    model_patcher = None
    clip_target = None

    parameters = util.calculate_parameters(sd, "model.diffusion_model.")
    load_device = Device.get_torch_device()

    model_config = unet.model_config_from_unet(sd, "model.diffusion_model.")
    unet_dtype = unet.unet_dtype1(
        model_params=parameters,
        supported_dtypes=model_config.supported_inference_dtypes,
    )
    manual_cast_dtype = Device.unet_manual_cast(
        unet_dtype, load_device, model_config.supported_inference_dtypes
    )
    model_config.set_inference_dtype(unet_dtype, manual_cast_dtype)

    if output_model:
        inital_load_device = Device.unet_inital_load_device(parameters, unet_dtype)
        offload_device = Device.unet_offload_device()
        model = model_config.get_model(
            sd, "model.diffusion_model.", device=inital_load_device
        )
        model.load_model_weights(sd, "model.diffusion_model.")

    if output_vae:
        vae_sd = util.state_dict_prefix_replace(
            sd, {k: "" for k in model_config.vae_key_prefix}, filter_keys=True
        )
        vae_sd = model_config.process_vae_state_dict(vae_sd)
        vae = VariationalAE.VAE(sd=vae_sd)

    if output_clip:
        clip_target = model_config.clip_target()
        if clip_target is not None:
            clip_sd = model_config.process_clip_state_dict(sd)
            if len(clip_sd) > 0:
                clip = Clip.CLIP(clip_target, embedding_directory=embedding_directory)
                m, u = clip.load_sd(clip_sd, full_model=True)
                if len(m) > 0:
                    m_filter = list(
                        filter(
                            lambda a: ".logit_scale" not in a
                            and ".transformer.text_projection.weight" not in a,
                            m,
                        )
                    )
                    if len(m_filter) > 0:
                        logging.warning("clip missing: {}".format(m))
                    else:
                        logging.debug("clip missing: {}".format(m))

                if len(u) > 0:
                    logging.debug("clip unexpected {}:".format(u))
            else:
                logging.warning(
                    "no CLIP/text encoder weights in checkpoint, the text encoder model will not be loaded."
                )

    left_over = sd.keys()
    if len(left_over) > 0:
        logging.debug("left over keys: {}".format(left_over))

    if output_model:
        model_patcher = ModelPatcher.ModelPatcher(
            model,
            load_device=load_device,
            offload_device=Device.unet_offload_device(),
            current_device=inital_load_device,
        )
        if inital_load_device != torch.device("cpu"):
            logging.info("loaded straight to GPU")
            Device.load_model_gpu(model_patcher)

    return (model_patcher, clip, vae, clipvision)

class CheckpointLoaderSimple:
    def load_checkpoint(self, ckpt_name, output_vae=True, output_clip=True):
        ckpt_path = f"{ckpt_name}"
        out = load_checkpoint_guess_config(
            ckpt_path,
            output_vae=True,
            output_clip=True,
            embedding_directory="./_internal/embeddings/",
        )
        print("loading", ckpt_path)
        return out[:3]