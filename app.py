import glob
import gradio as gr
import sys
import os
from PIL import Image
import spaces
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.user.pipeline import pipeline
import torch

def load_generated_images():
    """Load generated images with given prefix from disk"""
    image_files = glob.glob("./_internal/output/*")

    # If there are no image files, return
    if not image_files:
        return []

    # Sort files by modification time in descending order
    image_files.sort(key=os.path.getmtime, reverse=True)

    # Get most recent timestamp
    latest_time = os.path.getmtime(image_files[0])

    # Get all images from same batch (within 1 second of most recent)
    batch_images = []
    for file in image_files:
        if abs(os.path.getmtime(file) - latest_time) < 1.0:
            try:
                img = Image.open(file)
                batch_images.append(img)
            except:
                continue

    if not batch_images:
        return []
    return batch_images

@spaces.GPU
def generate_images(
    prompt: str,
    width: int = 512,
    height: int = 512,
    num_images: int = 1,
    batch_size: int = 1,
    hires_fix: bool = False,
    adetailer: bool = False,
    enhance_prompt: bool = False,
    img2img_enabled: bool = False,
    img2img_image: str = None,
    stable_fast: bool = False,
    reuse_seed: bool = False,
    flux_enabled: bool = False,
    prio_speed: bool = False,
    progress=gr.Progress()
):
    """Generate images using the LightDiffusion pipeline"""
    try:
        if img2img_enabled and img2img_image is not None:
            # Save uploaded image temporarily and pass path to pipeline
            img2img_image.save("temp_img2img.png")
            prompt = "temp_img2img.png"
        
        # Run pipeline and capture saved images
        with torch.inference_mode():
            images = pipeline(
                prompt=prompt,
                w=width,
                h=height, 
                number=num_images,
                batch=batch_size,
                hires_fix=hires_fix,
                adetailer=adetailer,
                enhance_prompt=enhance_prompt,
                img2img=img2img_enabled,
                stable_fast=stable_fast,
                reuse_seed=reuse_seed,
                flux_enabled=flux_enabled,
                prio_speed=prio_speed
            )
            
        return load_generated_images()

    except Exception as e:
        import traceback
        print(traceback.format_exc())
        return [Image.new('RGB', (512, 512), color='black')]

# Create Gradio interface
with gr.Blocks(title="LightDiffusion Web UI") as demo:
    gr.Markdown("# LightDiffusion Web UI")
    gr.Markdown("Generate AI images using LightDiffusion")
    
    with gr.Row():
        with gr.Column():
            # Input components
            prompt = gr.Textbox(label="Prompt", placeholder="Enter your prompt here...")
            
            with gr.Row():
                width = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Width")
                height = gr.Slider(minimum=64, maximum=2048, value=512, step=64, label="Height")
            
            with gr.Row():
                num_images = gr.Slider(minimum=1, maximum=10, value=1, step=1, label="Number of Images")
                batch_size = gr.Slider(minimum=1, maximum=4, value=1, step=1, label="Batch Size")
            
            with gr.Row():
                hires_fix = gr.Checkbox(label="HiRes Fix")
                adetailer = gr.Checkbox(label="Auto Face/Body Enhancement")
                enhance_prompt = gr.Checkbox(label="Enhance Prompt")
                stable_fast = gr.Checkbox(label="Stable Fast Mode")
            
            with gr.Row():
                reuse_seed = gr.Checkbox(label="Reuse Seed")
                flux_enabled = gr.Checkbox(label="Flux Mode") 
                prio_speed = gr.Checkbox(label="Prioritize Speed")
            
            with gr.Row():
                img2img_enabled = gr.Checkbox(label="Image to Image Mode")
                img2img_image = gr.Image(label="Input Image for img2img", visible=False)
            
            # Make input image visible only when img2img is enabled
            img2img_enabled.change(
                fn=lambda x: gr.update(visible=x),
                inputs=[img2img_enabled],
                outputs=[img2img_image]
            )
            
            generate_btn = gr.Button("Generate")

        # Output gallery
        gallery = gr.Gallery(
            label="Generated Images",
            show_label=True,
            elem_id="gallery",
            columns=[2], 
            rows=[2],
            object_fit="contain",
            height="auto"
        )
    
    # Connect generate button to pipeline
    generate_btn.click(
        fn=generate_images,
        inputs=[
            prompt,
            width, 
            height,
            num_images,
            batch_size,
            hires_fix,
            adetailer, 
            enhance_prompt,
            img2img_enabled,
            img2img_image,
            stable_fast,
            reuse_seed,
            flux_enabled,
            prio_speed
        ],
        outputs=gallery
    )

def is_huggingface_space():
    return "SPACE_ID" in os.environ

# For local testing
if __name__ == "__main__":
    if is_huggingface_space():
        demo.launch(
            debug=False,
            server_name="0.0.0.0",
            server_port=7860  # Standard HF Spaces port
        )
    else:
        demo.launch(
            server_name="0.0.0.0", 
            server_port=8000,
            auth=None,
            share=True,  # Only enable sharing locally
            debug=True
        )