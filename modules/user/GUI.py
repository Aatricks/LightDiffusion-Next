import os
import sys
import random
import threading
import tkinter as tk
from tkinter import filedialog
from PIL import Image, ImageTk
import numpy as np
import customtkinter as ctk
import glob

import torch


# Add the directory containing LightDiffusion.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.AutoDetailer import SAM, ADetailer, bbox, SEGS
from modules.AutoEncoders import VariationalAE
from modules.clip import Clip
from modules.sample import sampling

from modules.Utilities import util
from modules.UltimateSDUpscale import USDU_upscaler, UltimateSDUpscale as USDU

from modules.FileManaging import Downloader, ImageSaver, Loader
from modules.Model import LoRas
from modules.Utilities import Enhancer, Latent, upscale

Downloader.CheckAndDownload()

files = glob.glob("./_internal/checkpoints/*.safetensors")
loras = glob.glob("./_internal/loras/*.safetensors")
loras += glob.glob("./_internal/loras/*.pt")


class App(tk.Tk):
    """Main application class for the LightDiffusion GUI."""

    def __init__(self):
        """Initialize the App class."""
        super().__init__()
        self.title("LightDiffusion")
        self.geometry("800x700")

        file_names = [os.path.basename(file) for file in files]
        lora_names = [os.path.basename(lora) for lora in loras]

        selected_file = tk.StringVar()
        selected_lora = tk.StringVar()
        if file_names:
            selected_file.set(file_names[0])
        if lora_names:
            selected_lora.set(lora_names[0])

        # Create a frame for the sidebar
        self.sidebar = tk.Frame(self, width=300, bg="black")
        self.sidebar.pack(side=tk.LEFT, fill=tk.Y)

        # Text input for the prompt
        self.prompt_entry = ctk.CTkTextbox(self.sidebar, height=200, width=300)
        self.prompt_entry.pack(pady=10, padx=10)

        self.neg = ctk.CTkTextbox(self.sidebar, height=50, width=300)
        self.neg.pack(pady=10, padx=10)

        self.dropdown = ctk.CTkOptionMenu(self.sidebar, values=file_names)
        self.dropdown.pack()

        self.lora_selection = ctk.CTkOptionMenu(self.sidebar, values=lora_names)
        self.lora_selection.pack(pady=10)

        # Sliders for the resolution
        self.width_label = ctk.CTkLabel(self.sidebar, text="")
        self.width_label.pack()
        self.width_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=2048, number_of_steps=16
        )
        self.width_slider.pack()

        self.height_label = ctk.CTkLabel(self.sidebar, text="")
        self.height_label.pack()
        self.height_slider = ctk.CTkSlider(
            self.sidebar,
            from_=1,
            to=2048,
            number_of_steps=16,
        )
        self.height_slider.pack()

        self.cfg_label = ctk.CTkLabel(self.sidebar, text="")
        self.cfg_label.pack()
        self.cfg_slider = ctk.CTkSlider(
            self.sidebar, from_=1, to=15, number_of_steps=14
        )
        self.cfg_slider.pack()

        # Create a frame for the checkboxes
        self.checkbox_frame = tk.Frame(self.sidebar, bg="black")
        self.checkbox_frame.pack(pady=10)

        # checkbox for hiresfix
        self.hires_fix_var = tk.BooleanVar()
        self.hires_fix_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Hires Fix",
            variable=self.hires_fix_var,
            command=self.print_hires_fix,
        )
        self.hires_fix_checkbox.grid(row=0, column=0, padx=5, pady=5)

        # add a checkbox for Adetailer
        self.adetailer_var = tk.BooleanVar()
        self.adetailer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Adetailer",
            variable=self.adetailer_var,
            command=self.print_adetailer,
        )
        self.adetailer_checkbox.grid(row=0, column=1, padx=5, pady=5)

        # add a checkbox to enable stable-fast optimization
        self.stable_fast_var = tk.BooleanVar()
        self.stable_fast_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Stable Fast",
            variable=self.stable_fast_var,
        )
        self.stable_fast_checkbox.grid(row=1, column=0, padx=5, pady=5)

        # add a checkbox to enable prompt enhancer
        self.enhancer_var = tk.BooleanVar()
        self.enhancer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Prompt enhancer",
            variable=self.enhancer_var,
        )
        self.enhancer_checkbox.grid(row=1, column=1, padx=5, pady=5)

        # Button to launch the generation
        self.generate_button = ctk.CTkButton(
            self.sidebar, text="Generate", command=self.generate_image
        )
        self.generate_button.pack(pady=10)

        # Create a frame for the image display, without border
        self.display = tk.Frame(self, bg="black", border=0)
        self.display.pack(side=tk.RIGHT, expand=True, fill=tk.BOTH)

        # centered Label to display the generated image
        self.image_label = tk.Label(self.display, bg="black")
        self.image_label.pack(expand=True, padx=10, pady=10)

        self.previewer_var = tk.BooleanVar()
        self.previewer_checkbox = ctk.CTkCheckBox(
            self.display, text="Previewer", variable=self.previewer_var, command=self.print_previewer
        )
        self.previewer_checkbox.pack(pady=10)

        self.ckpt = None

        # load the checkpoint on an another thread
        threading.Thread(target=self._prep, daemon=True).start()

        self.button_frame = tk.Frame(self.sidebar, bg="black")
        self.button_frame.pack(pady=10)

        # add an img2img button, the button opens the file selector, run img2img on the selected image
        self.img2img_button = ctk.CTkButton(
            self.button_frame, text="img2img", command=self.img2img
        )
        self.img2img_button.grid(row=0, column=0, padx=5)

        self.interrupt_flag = False

        self.interrupt_button = ctk.CTkButton(
            self.button_frame, text="Interrupt", command=self.interrupt_generation
        )
        self.interrupt_button.grid(row=0, column=1, padx=5)

        prompt, neg, width, height, cfg = util.load_parameters_from_file()
        self.prompt_entry.insert(tk.END, prompt)
        self.neg.insert(tk.END, neg)
        self.width_slider.set(width)
        self.height_slider.set(height)
        self.cfg_slider.set(cfg)

        self.width_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.height_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.cfg_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.update_labels()
        self.prompt_entry.bind(
            "<KeyRelease>",
            lambda event: util.write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.neg.bind(
            "<KeyRelease>",
            lambda event: util.write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.width_slider.bind(
            "<ButtonRelease-1>",
            lambda event: util.write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.height_slider.bind(
            "<ButtonRelease-1>",
            lambda event: util.write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.cfg_slider.bind(
            "<ButtonRelease-1>",
            lambda event: util.write_parameters_to_file(
                self.prompt_entry.get("1.0", tk.END),
                self.neg.get("1.0", tk.END),
                self.width_slider.get(),
                self.height_slider.get(),
                self.cfg_slider.get(),
            ),
        )
        self.bind("<Configure>", self.on_resize)
        self.display_most_recent_image_flag = False
        self.display_most_recent_image()

    def _img2img(self, file_path: str) -> None:
        """Perform img2img on the selected image.

        Args:
            file_path (str): The path to the selected image.
        """
        self.display_most_recent_image_flag = False
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        img = Image.open(file_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            try:
                loraloader = LoRas.LoraLoader()
                loraloader_274 = loraloader.load_lora(
                    lora_name="add_detail.safetensors",
                    strength_model=2,
                    strength_clip=2,
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                )
            except:
                loraloader_274 = checkpointloadersimple_241

            if self.stable_fast_var.get() is True:
                from modules.StableFast import StableFast
                try:
                    app.title("LigtDiffusion - Generating StableFast model")
                except:
                    pass
                applystablefast = StableFast.ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                    enable_cuda_graph=False,
                    model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274

            clipsetlastlayer = Clip.CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )

            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            upscalemodelloader_244 = upscalemodelloader.load_model(
                "RealESRGAN_x4plus.pth"
            )
            try:
                app.title("LightDiffusion - Upscaling")
            except:
                pass
            ultimatesdupscale_250 = ultimatesdupscale.upscale(
                upscale_by=2,
                seed=random.randint(1, 2**64),
                steps=8,
                cfg=6,
                sampler_name="dpmpp_2m_sde",
                scheduler="karras",
                denoise=0.3,
                mode_type="Linear",
                tile_width=512,
                tile_height=512,
                mask_blur=16,
                tile_padding=32,
                seam_fix_mode="Half Tile",
                seam_fix_denoise=0.2,
                seam_fix_width=64,
                seam_fix_mask_blur=16,
                seam_fix_padding=32,
                force_uniform_tiles="enable",
                image=img_tensor,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                vae=checkpointloadersimple_241[2],
                upscale_model=upscalemodelloader_244[0],
            )
            saveimage.save_images(
                filename_prefix="LD-i2i",
                images=ultimatesdupscale_250[0],
            )
            for image in ultimatesdupscale_250[0]:
                i = 255.0 * image.cpu().numpy()
                img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
        self.update_image(img)
        global generated
        generated = img
        self.display_most_recent_image_flag = True
        try:
            app.title("LightDiffusion")
        except:
            pass

    def img2img(self) -> None:
        """Open the file selector and run img2img on the selected image."""
        file_path = filedialog.askopenfilename()
        if file_path:
            threading.Thread(
                target=self._img2img, args=(file_path,), daemon=True
            ).start()

    def print_hires_fix(self) -> None:
        """Print the status of the hires fix checkbox."""
        if self.hires_fix_var.get() is True:
            print("Hires fix is ON")
        else:
            print("Hires fix is OFF")

    def print_adetailer(self) -> None:
        """Print the status of the adetailer checkbox."""
        if self.adetailer_var.get() is True:
            print("Adetailer is ON")
        else:
            print("Adetailer is OFF")
            
    def print_previewer(self) -> None:
        """Print the status of the previewer checkbox."""
        if self.previewer_var.get() is True:
            print("Previewer is ON")
        else:
            print("Previewer is OFF")

    def generate_image(self) -> None:
        """Start the image generation process."""
        threading.Thread(target=self._generate_image, daemon=True).start()

    def _prep(self) -> tuple:
        """Prepare the necessary components for image generation.

        Returns:
            tuple: The prepared components.
        """
        if self.dropdown.get() != self.ckpt:
            self.ckpt = self.dropdown.get()
            with torch.inference_mode():
                self.checkpointloadersimple = Loader.CheckpointLoaderSimple()
                self.checkpointloadersimple_241 = (
                    self.checkpointloadersimple.load_checkpoint(
                        ckpt_name="./_internal/checkpoints/" + self.ckpt
                    )
                )
                self.cliptextencode = Clip.CLIPTextEncode()
                self.emptylatentimage = Latent.EmptyLatentImage()
                self.ksampler_instance = sampling.KSampler2()
                self.vaedecode = VariationalAE.VAEDecode()
                self.saveimage = ImageSaver.SaveImage()
                self.latent_upscale = upscale.LatentUpscale()
                self.upscalemodelloader = USDU_upscaler.UpscaleModelLoader()
                self.ultimatesdupscale = USDU.UltimateSDUpscale()
        return (
            self.checkpointloadersimple_241,
            self.cliptextencode,
            self.emptylatentimage,
            self.ksampler_instance,
            self.vaedecode,
            self.saveimage,
            self.latent_upscale,
            self.upscalemodelloader,
            self.ultimatesdupscale,
        )

    def _generate_image(self) -> None:
        """Generate an image based on the provided prompt and settings."""
        self.display_most_recent_image_flag = False
        prompt = self.prompt_entry.get("1.0", tk.END)
        if self.enhancer_var.get() is True:
            prompt = Enhancer.enhance_prompt()
            while prompt is None:
                pass
        neg = self.neg.get("1.0", tk.END)
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        cfg = int(self.cfg_slider.get())
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
                saveimage,
                latentupscale,
                upscalemodelloader,
                ultimatesdupscale,
            ) = self._prep()
            try:
                loraloader = LoRas.LoraLoader()
                loraloader_274 = loraloader.load_lora(
                    lora_name=self.lora_selection.get().replace(
                        "./_internal/loras/", ""
                    ),
                    strength_model=0.7,
                    strength_clip=0.7,
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                )
                print(
                    "loading",
                    self.lora_selection.get().replace("./_internal/loras/", ""),
                )
            except:
                loraloader_274 = checkpointloadersimple_241
            try:
                samloader = SAM.SAMLoader()
                samloader_87 = samloader.load_model(
                    model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                )

                cliptextencode_124 = cliptextencode.encode(
                    text="royal, detailed, magnificient, beautiful, seducing",
                    clip=loraloader_274[1],
                )

                ultralyticsdetectorprovider = bbox.UltralyticsDetectorProvider()
                ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                    # model_name="face_yolov8m.pt"
                    model_name="person_yolov8m-seg.pt"
                )

                bboxdetectorsegs = bbox.BboxDetectorForEach()
                samdetectorcombined = SAM.SAMDetectorCombined()
                impactsegsandmask = SEGS.SegsBitwiseAndMask()
                detailerforeachdebug = ADetailer.DetailerForEachTest()
            except:
                pass
            clipsetlastlayer = Clip.CLIPSetLastLayer()
            clipsetlastlayer_257 = clipsetlastlayer.set_last_layer(
                stop_at_clip_layer=-2, clip=loraloader_274[1]
            )
            if self.stable_fast_var.get() is True:
                from modules.StableFast import StableFast
                try:
                    self.title("LightDiffusion - Generating StableFast model")
                except:
                    pass
                applystablefast = StableFast.ApplyStableFastUnet()
                applystablefast_158 = applystablefast.apply_stable_fast(
                    enable_cuda_graph=False,
                    model=loraloader_274[0],
                )
            else:
                applystablefast_158 = loraloader_274
            cliptextencode_242 = cliptextencode.encode(
                text=prompt,
                clip=clipsetlastlayer_257[0],
            )
            cliptextencode_243 = cliptextencode.encode(
                text=neg,
                clip=clipsetlastlayer_257[0],
            )
            emptylatentimage_244 = emptylatentimage.generate(
                width=w, height=h, batch_size=1
            )
            ksampler_239 = ksampler_instance.sample(
                seed=random.randint(1, 2**64),
                steps=40,
                cfg=cfg,
                sampler_name="dpm_adaptive",
                scheduler="karras",
                denoise=1,
                model=applystablefast_158[0],
                positive=cliptextencode_242[0],
                negative=cliptextencode_243[0],
                latent_image=emptylatentimage_244[0],
            )
            if self.hires_fix_var.get() is True:
                latentupscale_254 = latentupscale.upscale(
                    width=w * 2,
                    height=h * 2,
                    samples=ksampler_239[0],
                )
                ksampler_253 = ksampler_instance.sample(
                    seed=random.randint(1, 2**64),
                    steps=10,
                    cfg=8,
                    sampler_name="euler_ancestral",
                    scheduler="normal",
                    denoise=0.45,
                    model=applystablefast_158[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=latentupscale_254[0],
                )
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_253[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(
                    filename_prefix="LD-HiresFix", images=vaedecode_240[0]
                )
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            else:
                vaedecode_240 = vaedecode.decode(
                    samples=ksampler_239[0],
                    vae=checkpointloadersimple_241[2],
                )
                saveimage.save_images(filename_prefix="LD", images=vaedecode_240[0])
                for image in vaedecode_240[0]:
                    i = 255.0 * image.cpu().numpy()
                    img = Image.fromarray(np.clip(i, 0, 255).astype(np.uint8))
            if self.adetailer_var.get() is True:
                bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                    threshold=0.5,
                    dilation=10,
                    crop_factor=2,
                    drop_size=10,
                    labels="all",
                    bbox_detector=ultralyticsdetectorprovider_151[0],
                    image=vaedecode_240[0],
                )
                samdetectorcombined_139 = samdetectorcombined.doit(
                    detection_hint="center-1",
                    dilation=0,
                    threshold=0.93,
                    bbox_expansion=0,
                    mask_hint_threshold=0.7,
                    mask_hint_use_negative="False",
                    sam_model=samloader_87[0],
                    segs=bboxdetectorsegs_132,
                    image=vaedecode_240[0],
                )
                if samdetectorcombined_139[0] is None:
                    return
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132,
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
                    scheduler="karras",
                    denoise=0.5,
                    feather=5,
                    noise_mask=True,
                    force_inpaint=True,
                    wildcard="",
                    cycle=1,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    image=vaedecode_240[0],
                    segs=impactsegsandmask_152[0],
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                    vae=checkpointloadersimple_241[2],
                    positive=cliptextencode_124[0],
                    negative=cliptextencode_243[0],
                )
                saveimage.save_images(
                    filename_prefix="LD-refined",
                    images=detailerforeachdebug_145[0],
                )
                ultralyticsdetectorprovider = bbox.UltralyticsDetectorProvider()
                ultralyticsdetectorprovider_151 = ultralyticsdetectorprovider.doit(
                    model_name="face_yolov9c.pt"
                )
                bboxdetectorsegs_132 = bboxdetectorsegs.doit(
                    threshold=0.5,
                    dilation=10,
                    crop_factor=2,
                    drop_size=10,
                    labels="all",
                    bbox_detector=ultralyticsdetectorprovider_151[0],
                    image=detailerforeachdebug_145[0],
                )
                samdetectorcombined_139 = samdetectorcombined.doit(
                    detection_hint="center-1",
                    dilation=0,
                    threshold=0.93,
                    bbox_expansion=0,
                    mask_hint_threshold=0.7,
                    mask_hint_use_negative="False",
                    sam_model=samloader_87[0],
                    segs=bboxdetectorsegs_132,
                    image=detailerforeachdebug_145[0],
                )
                impactsegsandmask_152 = impactsegsandmask.doit(
                    segs=bboxdetectorsegs_132,
                    mask=samdetectorcombined_139[0],
                )
                detailerforeachdebug_145 = detailerforeachdebug.doit(
                    guide_size=512,
                    guide_size_for=False,
                    max_size=768,
                    seed=random.randint(1, 2**64),
                    steps=40,
                    cfg=6.5,
                    sampler_name="dpmpp_2m_sde",
                    scheduler="karras",
                    denoise=0.5,
                    feather=5,
                    noise_mask=True,
                    force_inpaint=True,
                    wildcard="",
                    cycle=1,
                    inpaint_model=False,
                    noise_mask_feather=20,
                    image=detailerforeachdebug_145[0],
                    segs=impactsegsandmask_152[0],
                    model=checkpointloadersimple_241[0],
                    clip=checkpointloadersimple_241[1],
                    vae=checkpointloadersimple_241[2],
                    positive=cliptextencode_124[0],
                    negative=cliptextencode_243[0],
                )
                saveimage.save_images(
                    filename_prefix="lD-2ndrefined",
                    images=detailerforeachdebug_145[0],
                )
        self.update_image(img)
        self.display_most_recent_image_flag = True

    def update_labels(self) -> None:
        """Update the labels for the sliders."""
        self.width_label.configure(text=f"Width: {int(self.width_slider.get())}")
        self.height_label.configure(text=f"Height: {int(self.height_slider.get())}")
        self.cfg_label.configure(text=f"CFG: {int(self.cfg_slider.get())}")

    def update_image(self, img: Image.Image) -> None:
        """Update the displayed image.

        Args:
            img (Image.Image): The image to display.
        """
        # Calculate the aspect ratio of the original image
        aspect_ratio = img.width / img.height

        # Determine the new dimensions while maintaining the aspect ratio
        label_width = int(4 * self.winfo_width() / 7)
        label_height = int(4 * self.winfo_height() / 7)

        if label_width / aspect_ratio <= label_height:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)

        # Resize the image to the new dimensions
        try :
            img = img.resize((new_width, new_height), Image.LANCZOS)
        except RecursionError:
            pass
        if self.display_most_recent_image_flag is False:
            self._update_image_label(img)

    def _update_image_label(self, img: Image.Image) -> None:
        """Update the image label with the provided image.

        Args:
            img (Image.Image): The image to display.
        """
        # Convert the PIL image to a Tkinter PhotoImage
        tk_image = ImageTk.PhotoImage(img)
        # Update the image label with the Tkinter PhotoImage
        self.image_label.config(image=tk_image)
        # Keep a reference to the image to prevent it from being garbage collected
        self.image_label.image = tk_image

    def display_most_recent_image(self) -> None:
        """Display the most recent image from the output directory."""
        # Get a list of all image files in the output directory
        image_files = glob.glob("./_internal/output/*")

        # If there are no image files, return
        if not image_files:
            return

        # Sort the files by modification time in descending order
        image_files.sort(key=os.path.getmtime, reverse=True)

        # Open the most recent image file
        img = Image.open(image_files[0])
        self.update_image(img)

    def on_resize(self, event: tk.Event) -> None:
        """Handle the window resize event.

        Args:
            event (tk.Event): The resize event.
        """
        if hasattr(self, "img"):
            self.update_image(self.img)

    def interrupt_generation(self) -> None:
        """Interrupt the image generation process."""
        self.interrupt_flag = True


if __name__ == "__main__":
    from modules.user.app_instance import app

    app.mainloop()
