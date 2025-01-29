import glob
from huggingface_hub import hf_hub_download


def CheckAndDownload():
    """#### Check and download all the necessary safetensors and checkpoints models"""
    if glob.glob("./_internal/checkpoints/*.safetensors") == []:

        hf_hub_download(
            repo_id="Meina/MeinaMix",
            filename="Meina V10 - baked VAE.safetensors",
            local_dir="./_internal/checkpoints/",
        )
        hf_hub_download(
            repo_id="Lykon/DreamShaper",
            filename="DreamShaper_8_pruned.safetensors",
            local_dir="./_internal/checkpoints/",
        )
    if glob.glob("./_internal/yolos/*.pt") == []:

        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="hand_yolov9c.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="face_yolov9c.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="person_yolov8m-seg.pt",
            local_dir="./_internal/yolos/",
        )
        hf_hub_download(
            repo_id="segments-arnaud/sam_vit_b",
            filename="sam_vit_b_01ec64.pth",
            local_dir="./_internal/yolos/",
        )
    if glob.glob("./_internal/ESRGAN/*.pth") == []:

        hf_hub_download(
            repo_id="lllyasviel/Annotators",
            filename="RealESRGAN_x4plus.pth",
            local_dir="./_internal/ESRGAN/",
        )
    if glob.glob("./_internal/loras/*.safetensors") == []:

        hf_hub_download(
            repo_id="EvilEngine/add_detail",
            filename="add_detail.safetensors",
            local_dir="./_internal/loras/",
        )
    if glob.glob("./_internal/embeddings/*.pt") == []:

        hf_hub_download(
            repo_id="EvilEngine/badhandv4",
            filename="badhandv4.pt",
            local_dir="./_internal/embeddings/",
        )
        # hf_hub_download(
        #     repo_id="segments-arnaud/sam_vit_b",
        #     filename="EasyNegative.safetensors",
        #     local_dir="./_internal/embeddings/",
        # )
    if glob.glob("./_internal/vae_approx/*.pth") == []:

        hf_hub_download(
            repo_id="madebyollin/taesd",
            filename="taesd_decoder.safetensors",
            local_dir="./_internal/vae_approx/",
        )

def CheckAndDownloadFlux():
    """#### Check and download all the necessary safetensors and checkpoints models for FLUX"""
    if glob.glob("./_internal/embeddings/*.pt") == []:
        hf_hub_download(
            repo_id="EvilEngine/badhandv4",
            filename="badhandv4.pt",
            local_dir="./_internal/embeddings",
        )
    if glob.glob("./_internal/unet/*.gguf") == []:

        hf_hub_download(
            repo_id="city96/FLUX.1-dev-gguf",
            filename="flux1-dev-Q8_0.gguf",
            local_dir="./_internal/unet",
        )
    if glob.glob("./_internal/clip/*.gguf") == []:

        hf_hub_download(
            repo_id="city96/t5-v1_1-xxl-encoder-gguf",
            filename="t5-v1_1-xxl-encoder-Q8_0.gguf",
            local_dir="./_internal/clip",
        )
        hf_hub_download(
            repo_id="comfyanonymous/flux_text_encoders",
            filename="clip_l.safetensors",
            local_dir="./_internal/clip",
        )
    if glob.glob("./_internal/vae/*.safetensors") == []:

        hf_hub_download(
            repo_id="black-forest-labs/FLUX.1-schnell",
            filename="ae.safetensors",
            local_dir="./_internal/vae",
        )

    if glob.glob("./_internal/vae_approx/*.pth") == []:

        hf_hub_download(
            repo_id="madebyollin/taef1",
            filename="diffusion_pytorch_model.safetensors",
            local_dir="./_internal/vae_approx/",
        )
