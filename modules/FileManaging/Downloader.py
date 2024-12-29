import glob


def CheckAndDownload():
    """#### Check and download all the necessary safetensors and checkpoints models"""
    if glob.glob("./_internal/checkpoints/*.safetensors") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="Meina/MeinaMix",
            filename="Meina V10 - baked VAE.safetensors",
            local_dir="./_internal/checkpoints/",
        )
    if glob.glob("./_internal/yolos/*.pt") == []:
        from huggingface_hub import hf_hub_download

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
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="lllyasviel/Annotators",
            filename="RealESRGAN_x4plus.pth",
            local_dir="./_internal/ESRGAN/",
        )
    if glob.glob("./_internal/loras/*.safetensors") == []:
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="EvilEngine/add_detail",
            filename="add_detail.safetensors",
            local_dir="./_internal/loras/",
        )
    if glob.glob("./_internal/embeddings/*.pt") == []:
        from huggingface_hub import hf_hub_download

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
        from huggingface_hub import hf_hub_download

        hf_hub_download(
            repo_id="madebyollin/taesd",
            filename="taesd_decoder.safetensors",
            local_dir="./_internal/vae_approx/",
        )
