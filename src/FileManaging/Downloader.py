import glob
import os
import shutil
from pathlib import Path

from huggingface_hub import hf_hub_download


def CheckAndDownloadFlux2():
    """Check and download Flux2 Klein model components if needed.
    
    Downloads:
    - Flux2 Klein diffusion model to ./include/diffusion_model/
    - Qwen3 4B text encoder to ./include/text_encoder/
    - Flux VAE to ./include/vae/
    """
    
    # Check for diffusion model
    diffusion_dir = "./include/diffusion_model/"
    os.makedirs(diffusion_dir, exist_ok=True)
    
    # Look for actual diffusion model files (safetensors, pt, pth) that match flux/klein names, including nested subdirectories
    diffusion_candidates = glob.glob(os.path.join(diffusion_dir, "**", "*flux*"), recursive=True) + glob.glob(os.path.join(diffusion_dir, "**", "*klein*"), recursive=True)
    flux2_models = [p for p in diffusion_candidates if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".safetensors", ".pt", ".pth")]
    if not flux2_models:
        print("Downloading Flux2 Klein diffusion model...")
        try:
            filename = "split_files/diffusion_models/flux-2-klein-4b.safetensors"
            path = hf_hub_download(
                repo_id="black-forest-labs/FLUX.2-klein-4B",
                filename="flux-2-klein-4b.safetensors",
                local_dir=diffusion_dir,
            )
            # If it downloaded into a subfolder, move it up
            if os.path.dirname(filename):
                target = os.path.join(diffusion_dir, os.path.basename(filename))
                if os.path.exists(path) and not os.path.exists(target):
                    shutil.move(path, target)
                    # Cleanup empty subfolders
                    subfolder = os.path.join(diffusion_dir, filename.split('/')[0])
                    if os.path.exists(subfolder):
                        shutil.rmtree(subfolder)
        except Exception as e:
            print(f"Could not download Flux2 diffusion model: {e}")
    
    # Check for text encoder
    text_encoder_dir = "./include/text_encoder/"
    os.makedirs(text_encoder_dir, exist_ok=True)
    
    # Look for actual text encoder model files (safetensors, pt, pth) matching qwen/klein, including nested subdirs
    qwen_candidates = glob.glob(os.path.join(text_encoder_dir, "**", "*qwen*"), recursive=True) + glob.glob(os.path.join(text_encoder_dir, "**", "*klein*"), recursive=True)
    qwen_models = [p for p in qwen_candidates if os.path.isfile(p) and os.path.splitext(p)[1].lower() in (".safetensors", ".pt", ".pth")]
    if not qwen_models:
        print("Downloading Qwen3 text encoder for Flux2 Klein...")
        try:
            filename = "split_files/text_encoders/qwen_3_4b.safetensors"
            path = hf_hub_download(
                repo_id="Comfy-Org/vae-text-encorder-for-flux-klein-4b",
                filename=filename, 
                local_dir=text_encoder_dir,
            )
            # If it downloaded into a subfolder, move it up
            if os.path.dirname(filename):
                target = os.path.join(text_encoder_dir, os.path.basename(filename))
                if os.path.exists(path) and not os.path.exists(target):
                    shutil.move(path, target)
                    # Cleanup empty subfolders
                    subfolder = os.path.join(text_encoder_dir, filename.split('/')[0])
                    if os.path.exists(subfolder):
                        shutil.rmtree(subfolder)
        except Exception as e:
            print(f"Could not download Qwen text encoder: {e}")
    
    # Check for VAE
    vae_dir = "./include/vae/"
    os.makedirs(vae_dir, exist_ok=True)
    
    vae_models = glob.glob(f"{vae_dir}*.safetensors") + glob.glob(f"{vae_dir}*.pt")
    if not vae_models:
        print("Downloading Flux VAE...")
        try:
            filename = "split_files/vae/flux2-vae.safetensors"
            path = hf_hub_download(
                repo_id="Comfy-Org/flux2-dev",
                filename=filename,
                local_dir=vae_dir,
            )
            # If it downloaded into a subfolder, move it up
            if os.path.dirname(filename):
                target = os.path.join(vae_dir, os.path.basename(filename))
                if os.path.exists(path) and not os.path.exists(target):
                    shutil.move(path, target)
                    # Cleanup empty subfolders
                    subfolder = os.path.join(vae_dir, filename.split('/')[0])
                    if os.path.exists(subfolder):
                        shutil.rmtree(subfolder)
        except Exception as e:
            print(f"Could not download Flux VAE: {e}")


def CheckAndDownload():
    """#### Check and download all the necessary safetensors and checkpoints models"""
    if glob.glob("./include/checkpoints/*.safetensors") == []:
        hf_hub_download(
            repo_id="Lykon/DreamShaper",
            filename="DreamShaper_8_pruned.safetensors",
            local_dir="./include/checkpoints/",
        )
    if glob.glob("./include/yolos/*.pt") == []:

        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="hand_yolov9c.pt",
            local_dir="./include/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="face_yolov9c.pt",
            local_dir="./include/yolos/",
        )
        hf_hub_download(
            repo_id="Bingsu/adetailer",
            filename="person_yolov8m-seg.pt",
            local_dir="./include/yolos/",
        )
        hf_hub_download(
            repo_id="segments-arnaud/sam_vit_b",
            filename="sam_vit_b_01ec64.pth",
            local_dir="./include/yolos/",
        )
    if glob.glob("./include/ESRGAN/*.pth") == []:

        hf_hub_download(
            repo_id="lllyasviel/Annotators",
            filename="RealESRGAN_x4plus.pth",
            local_dir="./include/ESRGAN/",
        )
    if glob.glob("./include/loras/*.safetensors") == []:

        hf_hub_download(
            repo_id="EvilEngine/add_detail",
            filename="add_detail.safetensors",
            local_dir="./include/loras/",
        )
    if glob.glob("./include/embeddings/*.pt") == []:

        hf_hub_download(
            repo_id="EvilEngine/badhandv4",
            filename="badhandv4.pt",
            local_dir="./include/embeddings/",
        )
        # hf_hub_download(
        #     repo_id="segments-arnaud/sam_vit_b",
        #     filename="EasyNegative.safetensors",
        #     local_dir="./include/embeddings/",
        # )
    if glob.glob("./include/vae_approx/taesd_decoder.safetensors") == []:

        hf_hub_download(
            repo_id="madebyollin/taesd",
            filename="taesd_decoder.safetensors",
            local_dir="./include/vae_approx/",
        )
    
    # Check and download Flux2 components if user wants to use them
    # This is optional - only downloads if the directories don't exist
    # Users can manually trigger by calling CheckAndDownloadFlux2()
