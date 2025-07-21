import os
import queue
import sys
import random
import threading
import tkinter as tk
from tkinter import filedialog
from typing import Union
from PIL import Image, ImageTk
import numpy as np
import customtkinter as ctk
import glob
import time

import torch

# Add the directory containing LightDiffusion.py to the Python path
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "../..")))

from modules.AutoDetailer import SAM, ADetailer, bbox, SEGS
from modules.AutoEncoders import VariationalAE
from modules.clip import Clip
from modules.sample import sampling

from modules.Utilities import util
from modules.UltimateSDUpscale import USDU_upscaler, UltimateSDUpscale

from modules.FileManaging import Downloader, ImageSaver, Loader
from modules.Model import LoRas
from modules.Utilities import Enhancer, Latent, upscale
from modules.Quantize import Quantizer
from modules.WaveSpeed import fbcache_nodes
from modules.hidiffusion import msw_msa_attention
from modules.AutoHDR import ahdr

Downloader.CheckAndDownload()

files = glob.glob("./_internal/checkpoints/*.safetensors")
loras = glob.glob("./_internal/loras/*.safetensors")
loras += glob.glob("./_internal/loras/*.pt")


def debounce(wait):
    """Decorator to debounce resize events"""

    def decorator(fn):
        last_call = [0]

        def debounced(*args, **kwargs):
            current_time = time.time()
            if current_time - last_call[0] >= wait:
                fn(*args, **kwargs)
                last_call[0] = current_time

        return debounced

    return decorator


class App(tk.Tk):
    """Main application class for the LightDiffusion GUI."""

    def __init__(self):
        """Initialize the App class."""
        super().__init__()
        self.title("LightDiffusion")
        self.geometry("900x800")

        # Configure main window grid
        self.grid_columnconfigure(1, weight=1)
        self.grid_rowconfigure(0, weight=1)

        file_names = [os.path.basename(file) for file in files]
        lora_names = [os.path.basename(lora) for lora in loras]

        selected_file = tk.StringVar()
        selected_lora = tk.StringVar()
        if file_names:
            selected_file.set(file_names[0])
        if lora_names:
            selected_lora.set(lora_names[0])

        # Create main sidebar frame with padding and grid
        self.sidebar = tk.Frame(self, bg="#FBFBFB", padx=10, pady=10)
        self.sidebar.grid(row=0, column=0, sticky="nsew")
        self.sidebar.grid_columnconfigure(0, weight=1)

        # Configure sidebar grid rows
        for i in range(8):
            self.sidebar.grid_rowconfigure(i, weight=1)

        # Text input frames with expansion
        self.prompt_frame = tk.Frame(self.sidebar, bg="#FBFBFB")
        self.prompt_frame.grid(row=0, column=0, sticky="nsew", pady=(0, 5))
        self.prompt_frame.grid_columnconfigure(0, weight=1)
        self.prompt_frame.grid_rowconfigure(0, weight=2)
        self.prompt_frame.grid_rowconfigure(1, weight=1)

        # Prompt textbox with expansion
        self.prompt_entry = ctk.CTkTextbox(
            self.prompt_frame,
            height=150,
            fg_color="#E8F9FF",
            text_color="black",
            border_color="gray",
            border_width=2,
        )
        self.prompt_entry.grid(row=0, column=0, sticky="nsew")

        # Negative prompt textbox with expansion
        self.neg = ctk.CTkTextbox(
            self.prompt_frame,
            height=75,
            fg_color="#E8F9FF",
            text_color="black",
            border_color="gray",
            border_width=2,
        )
        self.neg.grid(row=1, column=0, sticky="nsew", pady=(5, 0))

        # Add model dropdown with error handling for empty lists
        model_values = (file_names if file_names else ["No models found"]) + ["flux"]

        # Model dropdown and Flux checkbox
        self.dropdown = ctk.CTkOptionMenu(
            self.sidebar,
            values=model_values,
            fg_color="#F5EFFF",
            text_color="black",
            command=self.on_model_selected,
        )
        self.dropdown.grid(row=2, column=0, sticky="ew")

        # LoRA selection
        self.lora_selection = ctk.CTkOptionMenu(
            self.sidebar, values=lora_names, fg_color="#F5EFFF", text_color="black"
        )
        self.lora_selection.grid(row=3, column=0, sticky="ew", pady=5)

        # Display frame with expansion
        self.display = tk.Frame(self, bg="#FBFBFB")
        self.display.grid(row=0, column=1, sticky="nsew", padx=10, pady=10)
        self.display.grid_columnconfigure(0, weight=1)
        self.img = None

        # Add row configuration for both image and checkbox
        self.display.grid_rowconfigure(0, weight=1)  # For image
        self.display.grid_rowconfigure(1, weight=0)  # For checkbox

        # Image label with expansion
        self.image_label = tk.Label(self.display, bg="#FBFBFB")
        self.image_label.grid(row=0, column=0, sticky="nsew")

        # Previewer checkbox - changed from pack to grid
        self.previewer_var = tk.BooleanVar()
        self.previewer_checkbox = ctk.CTkCheckBox(
            self.display,
            text="Previewer",
            variable=self.previewer_var,
            command=self.print_previewer,
            text_color="black",
        )
        self.previewer_checkbox.grid(row=1, column=0, pady=10)

        # Progress Bar
        self.progress = ctk.CTkProgressBar(self.display, fg_color="#FBFBFB")
        self.progress.grid(row=2, column=0, sticky="ew", pady=10, padx=10)
        self.progress.set(0)

        # Make sliders frame expand
        self.sliders_frame = tk.Frame(self.sidebar, bg="#FBFBFB")
        self.sliders_frame.grid(row=4, column=0, sticky="nsew", pady=5)
        self.sliders_frame.grid_columnconfigure(1, weight=1)

        # Configure slider weights
        for i in range(3):
            self.sliders_frame.grid_rowconfigure(i, weight=1)

        # Make checkbox frame expand
        self.checkbox_frame = tk.Frame(self.sidebar, bg="#FBFBFB")
        self.checkbox_frame.grid(row=5, column=0, sticky="nsew", pady=10)
        self.checkbox_frame.grid_columnconfigure(0, weight=1)
        self.checkbox_frame.grid_columnconfigure(1, weight=1)

        # Make button frame expand
        self.button_frame = tk.Frame(self.sidebar, bg="#FBFBFB")
        self.button_frame.grid(row=7, column=0, sticky="nsew", pady=10)
        self.button_frame.grid_columnconfigure(0, weight=1)
        self.button_frame.grid_columnconfigure(1, weight=1)

        # Width slider
        tk.Label(self.sliders_frame, text="Width:", bg="#FBFBFB").grid(
            row=0, column=0, padx=(0, 5)
        )
        self.width_slider = ctk.CTkSlider(
            self.sliders_frame, from_=1, to=2048, number_of_steps=32, fg_color="#F5EFFF"
        )
        self.width_slider.grid(row=0, column=1, sticky="ew")
        self.width_label = ctk.CTkLabel(self.sliders_frame, text="")
        self.width_label.grid(row=0, column=2, padx=(5, 0))

        # Height slider
        tk.Label(self.sliders_frame, text="Height:", bg="#FBFBFB").grid(
            row=1, column=0, padx=(0, 5)
        )
        self.height_slider = ctk.CTkSlider(
            self.sliders_frame, from_=1, to=2048, number_of_steps=32, fg_color="#F5EFFF"
        )
        self.height_slider.grid(row=1, column=1, sticky="ew")
        self.height_label = ctk.CTkLabel(self.sliders_frame, text="")
        self.height_label.grid(row=1, column=2, padx=(5, 0))

        # CFG slider
        tk.Label(self.sliders_frame, text="CFG:", bg="#FBFBFB").grid(
            row=2, column=0, padx=(0, 5)
        )
        self.cfg_slider = ctk.CTkSlider(
            self.sliders_frame, from_=1, to=15, number_of_steps=14, fg_color="#F5EFFF"
        )
        self.cfg_slider.grid(row=2, column=1, sticky="ew")
        self.cfg_label = ctk.CTkLabel(self.sliders_frame, text="")
        self.cfg_label.grid(row=2, column=2, padx=(5, 0))

        # Batch size slider
        tk.Label(self.sliders_frame, text="Batch Size:", bg="#FBFBFB").grid(
            row=3, column=0, padx=(0, 5)
        )
        self.batch_slider = ctk.CTkSlider(
            self.sliders_frame, from_=1, to=10, number_of_steps=9, fg_color="#F5EFFF"
        )
        self.batch_slider.grid(row=3, column=1, sticky="ew")
        self.batch_label = ctk.CTkLabel(self.sliders_frame, text="")
        self.batch_label.grid(row=3, column=2, padx=(5, 0))

        # Configure grid columns and rows to distribute space evenly
        self.checkbox_frame.grid_columnconfigure(0, weight=1)
        self.checkbox_frame.grid_columnconfigure(1, weight=1)
        self.checkbox_frame.grid_rowconfigure(0, weight=1)
        self.checkbox_frame.grid_rowconfigure(1, weight=1)
        self.checkbox_frame.grid_rowconfigure(2, weight=1)
        self.checkbox_frame.grid_rowconfigure(3, weight=1)

        # checkbox for hiresfix
        self.hires_fix_var = tk.BooleanVar()
        self.hires_fix_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Hires Fix",
            variable=self.hires_fix_var,
            command=self.print_hires_fix,
            text_color="black",
        )
        self.hires_fix_checkbox.grid(
            row=0, column=0, padx=(75, 5), pady=5, sticky="nsew"
        )

        # checkbox for Adetailer
        self.adetailer_var = tk.BooleanVar()
        self.adetailer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Adetailer",
            variable=self.adetailer_var,
            command=self.print_adetailer,
            text_color="black",
        )
        self.adetailer_checkbox.grid(row=0, column=1, padx=5, pady=5, sticky="nsew")

        # checkbox to enable stable-fast optimization
        self.stable_fast_var = tk.BooleanVar()
        self.stable_fast_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Stable Fast",
            variable=self.stable_fast_var,
            text_color="black",
        )
        self.stable_fast_checkbox.grid(
            row=1, column=0, padx=(75, 5), pady=5, sticky="nsew"
        )

        # checkbox to enable prompt enhancer
        self.enhancer_var = tk.BooleanVar()
        self.enhancer_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Prompt enhancer",
            variable=self.enhancer_var,
            text_color="black",
        )
        self.enhancer_checkbox.grid(row=1, column=1, padx=5, pady=5, sticky="nsew")

        self.prioritize_speed_var = tk.BooleanVar()
        self.prioritize_speed_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Prioritize Speed",
            variable=self.prioritize_speed_var,
            text_color="black",
        )
        self.prioritize_speed_checkbox.grid(
            row=2, column=0, padx=(75, 5), pady=5, sticky="nsew"
        )

        # checkbox to enable multi-scale diffusion
        self.multiscale_var = tk.BooleanVar(value=True)
        self.multiscale_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Multi-Scale",
            variable=self.multiscale_var,
            text_color="black",
        )
        self.multiscale_checkbox.grid(row=2, column=1, padx=5, pady=5, sticky="nsew")

        # checkbox to enable intermittent full-res
        self.multiscale_intermittent_var = tk.BooleanVar(value=True)
        self.multiscale_intermittent_checkbox = ctk.CTkCheckBox(
            self.checkbox_frame,
            text="Intermittent Full-Res",
            variable=self.multiscale_intermittent_var,
            text_color="black",
        )
        self.multiscale_intermittent_checkbox.grid(
            row=3, column=0, padx=(75, 5), pady=5, sticky="nsew"
        )

        # Button to launch the generation
        self.generate_button = ctk.CTkButton(
            self.sidebar,
            text="Generate",
            command=self.generate_image,
            fg_color="#C4D9FF",
            text_color="black",
            border_color="gray",
            border_width=2,
        )
        self.generate_button.grid(
            row=6, column=0, pady=10, sticky="ew"
        )  # Changed from pack to grid

        self.ckpt = None

        # load the checkpoint on an another thread
        threading.Thread(target=self._prep, daemon=True).start()

        # img2img button
        self.img2img_button = ctk.CTkButton(
            self.button_frame,
            text="img2img",
            command=self.img2img,
            fg_color="#F5EFFF",
            text_color="black",
            border_color="gray",
            border_width=2,
        )
        self.img2img_button.grid(row=0, column=0, padx=5, sticky="ew")

        # interrupt button
        self.generation_threads = []
        self.interrupt_flag = False
        self.interrupt_button = ctk.CTkButton(
            self.button_frame,
            text="Interrupt",
            command=self.interrupt_generation,
            fg_color="#F5EFFF",
            text_color="black",
            border_color="gray",
            border_width=2,
        )
        self.interrupt_button.grid(row=0, column=1, padx=5, sticky="ew")

        prompt, neg, width, height, cfg = util.load_parameters_from_file()
        self.prompt_entry.insert(tk.END, prompt)
        self.neg.insert(tk.END, neg)
        self.width_slider.set(width)
        self.height_slider.set(height)
        self.cfg_slider.set(cfg)
        self.batch_slider.set(1)

        self.width_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.height_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.cfg_slider.bind("<B1-Motion>", lambda event: self.update_labels())
        self.batch_slider.bind("<B1-Motion>", lambda event: self.update_labels())
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
        # Add resize handling variables
        self._resize_queue = queue.Queue()
        self._resize_thread = None
        self._resize_event = threading.Event()
        self._resize_lock = threading.Lock()
        self._resize_running = True
        self._last_resize_time = 0
        self._resize_delay = 0.1
        self._image_cache = {}
        self._current_image = None

        # Start resize worker thread
        self._start_resize_worker()

        # Bind resize event
        self.bind("<Configure>", self._queue_resize)

        # Bind cleanup
        self.protocol("WM_DELETE_WINDOW", self._cleanup)
        self.display_most_recent_image_flag = False
        self.display_most_recent_image()
        self.is_generating = False
        self.sampler = (
            "dpmpp_sde_cfgpp"
            if not self.prioritize_speed_var.get()
            else "dpmpp_2m_cfgpp"
        )

    def _img2img(self, file_path: str) -> None:
        """Perform img2img on the selected image.

        Args:
            file_path (str): The path to the selected image.
        """
        self.is_generating = True
        self.img2img_button.configure(state="disabled")
        self.display_most_recent_image_flag = False
        prompt = self.prompt_entry.get("1.0", tk.END)
        neg = self.neg.get("1.0", tk.END)
        img = Image.open(file_path)
        img_array = np.array(img)
        img_tensor = torch.from_numpy(img_array).float().to("cpu") / 255.0
        img_tensor = img_tensor.unsqueeze(0)
        self.interrupt_flag = False
        self.sampler = (
            "dpmpp_sde_cfgpp"
            if not self.prioritize_speed_var.get()
            else "dpmpp_2m_cfgpp"
        )
        with torch.inference_mode():
            (
                checkpointloadersimple_241,
                cliptextencode,
                emptylatentimage,
                ksampler_instance,
                vaedecode,
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
                fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                applystablefast_158 = fb_cache.patch(
                    applystablefast_158, "diffusion_model", 0.120
                )
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
                sampler_name=self.sampler,
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
            self.update_from_decode(ultimatesdupscale_250[0], "LD-I2I")
        self.update_image(img)
        global generated
        generated = img
        self.display_most_recent_image_flag = True
        try:
            app.title("LightDiffusion")
        except:
            pass
        self.is_generating = False
        self.img2img_button.configure(state="normal")

    def img2img(self) -> None:
        """Open the file selector and run img2img on the selected image."""
        if self.is_generating:
            return
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
        if self.is_generating:
            return

        if self.dropdown.get() == "flux":
            self.generate_thread = threading.Thread(
                target=self._generate_image_flux, daemon=True
            ).start()
        else:
            self.generate_thread = threading.Thread(
                target=self._generate_image, daemon=True
            ).start()

    def _prep(self) -> tuple:
        """Prepare the necessary components for image generation.

        Returns:
            tuple: The prepared components.
        """
        if self.dropdown.get() != self.ckpt and self.dropdown.get() != "flux":
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
                self.ksampler_instance = sampling.KSampler()
                self.vaedecode = VariationalAE.VAEDecode()
                self.latent_upscale = upscale.LatentUpscale()
                self.upscalemodelloader = USDU_upscaler.UpscaleModelLoader()
                self.ultimatesdupscale = UltimateSDUpscale.UltimateSDUpscale()
        return (
            self.checkpointloadersimple_241,
            self.cliptextencode,
            self.emptylatentimage,
            self.ksampler_instance,
            self.vaedecode,
            self.latent_upscale,
            self.upscalemodelloader,
            self.ultimatesdupscale,
        )

    def _generate_image(self) -> None:
        """Generate image with proper interrupt handling."""
        self.is_generating = True
        self.generate_button.configure(state="disabled")

        current_thread = threading.current_thread()
        self.generation_threads.append(current_thread)
        self.interrupt_flag = False
        self.sampler = (
            "dpmpp_sde_cfgpp"
            if not self.prioritize_speed_var.get()
            else "dpmpp_2m_cfgpp"
        )
        try:
            # Disable generate button during generation
            self.generate_button.configure(state="disabled")
            self.display_most_recent_image_flag = False
            self.progress.set(0)

            # Early interrupt check
            if self.interrupt_flag:
                return

            # Get generation parameters
            prompt = self.prompt_entry.get("1.0", tk.END)
            neg = self.neg.get("1.0", tk.END)
            w = int(self.width_slider.get())
            h = int(self.height_slider.get())
            cfg = int(self.cfg_slider.get())

            try:
                if self.enhancer_var.get() is True:
                    prompt = Enhancer.enhance_prompt(prompt)
                    while prompt is None:
                        pass
            except:
                pass

            # Main generation with proper interrupt handling
            with torch.inference_mode():
                components = self._prep()
                if self.interrupt_flag:
                    return
                (
                    checkpointloadersimple_241,
                    cliptextencode,
                    emptylatentimage,
                    ksampler_instance,
                    vaedecode,
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
                self.progress.set(0.2)
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
                    fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                    applystablefast_158 = fb_cache.patch(
                        applystablefast_158, "diffusion_model", 0.120
                    )
                hidiffoptimizer = msw_msa_attention.ApplyMSWMSAAttentionSimple()
                cliptextencode_242 = cliptextencode.encode(
                    text=prompt,
                    clip=clipsetlastlayer_257[0],
                )
                cliptextencode_243 = cliptextencode.encode(
                    text=neg,
                    clip=clipsetlastlayer_257[0],
                )
                emptylatentimage_244 = emptylatentimage.generate(
                    width=w, height=h, batch_size=int(self.batch_slider.get())
                )
                ksampler_239 = ksampler_instance.sample(
                    seed=random.randint(1, 2**64),
                    steps=20,
                    cfg=cfg,
                    sampler_name=self.sampler,
                    scheduler="karras",
                    denoise=1,
                    model=hidiffoptimizer.go(
                        model_type="auto", model=applystablefast_158[0]
                    )[0],
                    positive=cliptextencode_242[0],
                    negative=cliptextencode_243[0],
                    latent_image=emptylatentimage_244[0],
                    enable_multiscale=self.multiscale_var.get(),
                    multiscale_factor=0.5,
                    multiscale_fullres_start=5,
                    multiscale_fullres_end=8,
                    multiscale_intermittent_fullres=self.multiscale_intermittent_var.get(),
                )
                self.progress.set(0.4)
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
                        sampler_name="euler_ancestral_cfgpp",
                        scheduler="normal",
                        denoise=0.45,
                        model=hidiffoptimizer.go(
                            model_type="auto", model=applystablefast_158[0]
                        )[0],
                        positive=cliptextencode_242[0],
                        negative=cliptextencode_243[0],
                        latent_image=latentupscale_254[0],
                    )
                    vaedecode_240 = vaedecode.decode(
                        samples=ksampler_253[0],
                        vae=checkpointloadersimple_241[2],
                    )
                    self.update_from_decode(vaedecode_240[0], "LD-HF")
                else:
                    vaedecode_240 = vaedecode.decode(
                        samples=ksampler_239[0],
                        vae=checkpointloadersimple_241[2],
                    )
                    self.update_from_decode(vaedecode_240[0], "LD")
                if self.interrupt_flag:
                    return

                self.progress.set(0.6)
                if self.adetailer_var.get() is True:
                    samloader = SAM.SAMLoader()
                    samloader_87 = samloader.load_model(
                        model_name="sam_vit_b_01ec64.pth", device_mode="AUTO"
                    )
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
                        steps=20,
                        cfg=6.5,
                        sampler_name=self.sampler,
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
                        model=applystablefast_158[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                    )
                    self.update_from_decode(detailerforeachdebug_145[0], "LD-body")
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
                        steps=20,
                        cfg=6.5,
                        sampler_name=self.sampler,
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
                        model=applystablefast_158[0],
                        clip=checkpointloadersimple_241[1],
                        vae=checkpointloadersimple_241[2],
                        positive=cliptextencode_124[0],
                        negative=cliptextencode_243[0],
                    )
                    self.update_from_decode(detailerforeachdebug_145[0], "LD-head")

                self.progress.set(0.8)

        except Exception as e:
            print(f"Generation error: {e}")
            self.title(f"LightDiffusion - Error: {str(e)}")

        finally:
            # Reset state when done
            self.is_generating = False
            self.generate_button.configure(state="normal")
            if current_thread in self.generation_threads:
                self.generation_threads.remove(current_thread)
            self.progress.set(0)

            # Clear CUDA cache
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    def _generate_image_flux(self) -> None:
        """Generate an image using the Flux model."""
        self.is_generating = True
        self.generate_button.configure(state="disabled")
        # Add current thread to list at start
        current_thread = threading.current_thread()
        self.generation_threads.append(current_thread)
        self.display_most_recent_image_flag = False
        w = int(self.width_slider.get())
        h = int(self.height_slider.get())
        prompt = self.prompt_entry.get("1.0", tk.END)
        try:
            if self.enhancer_var.get() is True:
                prompt = Enhancer.enhance_prompt(prompt)
                while prompt is None:
                    pass
            self.interrupt_flag = False
            Downloader.CheckAndDownloadFlux()
            with torch.inference_mode():
                dualcliploadergguf = Quantizer.DualCLIPLoaderGGUF()
                emptylatentimage = Latent.EmptyLatentImage()
                vaeloader = VariationalAE.VAELoader()
                unetloadergguf = Quantizer.UnetLoaderGGUF()
                cliptextencodeflux = Quantizer.CLIPTextEncodeFlux()
                conditioningzeroout = Quantizer.ConditioningZeroOut()
                ksampler = sampling.KSampler()
                vaedecode = VariationalAE.VAEDecode()
                unetloadergguf_10 = unetloadergguf.load_unet(
                    unet_name="flux1-dev-Q8_0.gguf"
                )
                vaeloader_11 = vaeloader.load_vae(vae_name="ae.safetensors")
                dualcliploadergguf_19 = dualcliploadergguf.load_clip(
                    clip_name1="clip_l.safetensors",
                    clip_name2="t5-v1_1-xxl-encoder-Q8_0.gguf",
                    type="flux",
                )
                emptylatentimage_5 = emptylatentimage.generate(
                    width=w, height=h, batch_size=int(self.batch_slider.get())
                )
                cliptextencodeflux_15 = cliptextencodeflux.encode(
                    clip_l=prompt,
                    t5xxl=prompt,
                    guidance=3.0,
                    clip=dualcliploadergguf_19[0],
                    flux_enabled=True,
                )
                conditioningzeroout_16 = conditioningzeroout.zero_out(
                    conditioning=cliptextencodeflux_15[0]
                )
                fb_cache = fbcache_nodes.ApplyFBCacheOnModel()
                unetloadergguf_10 = fb_cache.patch(
                    unetloadergguf_10, "diffusion_model", 0.120
                )
                # try:
                #     import triton
                #     compiler = misc_nodes.EnhancedCompileModel()
                #     unetloadergguf_10 = compiler.patch(unetloadergguf_10, True, "diffusion_model", "torch.compile", False, False, None, None, False, "inductor")
                # except ImportError:
                #     print("Triton not found, skipping compilation")
                ksampler_3 = ksampler.sample(
                    seed=random.randint(1, 2**64),
                    steps=20,
                    cfg=1,
                    sampler_name="euler_cfgpp",
                    scheduler="beta",
                    denoise=1,
                    model=unetloadergguf_10[0],
                    positive=cliptextencodeflux_15[0],
                    negative=conditioningzeroout_16[0],
                    latent_image=emptylatentimage_5[0],
                    flux=True,
                )
                vaedecode_8 = vaedecode.decode(
                    samples=ksampler_3[0],
                    vae=vaeloader_11[0],
                    flux=True,
                )
                self.update_from_decode(vaedecode_8[0], "LD-Flux")
        finally:
            # Reset state when done
            self.is_generating = False
            self.generate_button.configure(state="normal")
            if current_thread in self.generation_threads:
                self.generation_threads.remove(current_thread)

    def on_model_selected(self, *args):
        """Handle model selection changes"""
        if self.dropdown.get() == "flux":
            # Disable incompatible controls
            self.adetailer_checkbox._state = tk.DISABLED
            self.hires_fix_checkbox._state = tk.DISABLED
            self.stable_fast_checkbox._state = tk.DISABLED
            self.lora_selection._state = tk.DISABLED
            self.cfg_slider._state = tk.DISABLED
        else:
            # Enable controls
            self.adetailer_checkbox._state = tk.NORMAL
            self.hires_fix_checkbox._state = tk.NORMAL
            self.stable_fast_checkbox._state = tk.NORMAL
            self.lora_selection._state = tk.NORMAL
            self.cfg_slider._state = tk.NORMAL

    def _handle_decoded_image(self, decoded, prefix: str) -> None:
        """Handle decoded image processing with HDR effects.

        Args:
            decoded: Decoded tensor image
            prefix: Prefix for saved files
        """
        try:
            # Initialize components
            saveimage = ImageSaver.SaveImage()
            hdr = ahdr.HDREffects()
            images = []

            # Apply HDR effects
            if isinstance(decoded, tuple):
                # Handle tuple return
                tensor_image = decoded[0]
            else:
                tensor_image = decoded

            # Apply HDR as batch process
            processed = hdr.apply_hdr2(tensor_image)

            # Save images with prefix
            saveimage.save_images(
                filename_prefix=prefix,
                images=processed[0] if isinstance(processed, tuple) else processed,
            )

            # Convert processed tensors to PIL images
            for img_tensor in (
                processed[0] if isinstance(processed, tuple) else [processed]
            ):
                # Convert to numpy and scale
                img_array = 255.0 * img_tensor.cpu().numpy()

                # Handle different dimensions
                if img_array.ndim == 4:
                    img_array = np.squeeze(img_array)
                    img_array = img_array.reshape(
                        -1, img_array.shape[-2], img_array.shape[-1]
                    )

                # Convert to PIL image
                img = Image.fromarray(np.clip(img_array, 0, 255).astype(np.uint8))
                images.append(img)

            # Update display if not interrupted
            if not self.interrupt_flag:
                self.progress.set(1.0)
                if images:
                    self.img = images[0]
                    self.update_image(images)
                    self.display_most_recent_image_flag = True

        except Exception as e:
            print(f"Image processing error: {e}")
            self.title(f"LightDiffusion - Error: {str(e)}")

    def update_from_decode(self, decoded: Image.Image, prefix: str) -> None:
        """Update the image from the decode function.

        Args:
            decoded (Image.Image): The decoded image tensor/tuple
            prefix (str): Prefix for saved files
        """
        try:
            # Handle image processing in separate function
            self._handle_decoded_image(decoded, prefix)

        except Exception as e:
            print(f"Decode error: {e}")
            self.title(f"LightDiffusion - Error: {str(e)}")

        finally:
            # Ensure cleanup
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
                torch.cuda.synchronize()

    def update_labels(self) -> None:
        """Update the labels for the sliders."""
        self.width_label.configure(text=f"{int(self.width_slider.get())}")
        self.height_label.configure(text=f"{int(self.height_slider.get())}")
        self.cfg_label.configure(text=f"{int(self.cfg_slider.get())}")
        self.batch_label.configure(text=f"{int(self.batch_slider.get())}")

    def create_image_grid(self, images: list[Image.Image]) -> Image.Image:
        """Create a grid of images.

        Args:
            images (list[Image.Image]): List of images to arrange in grid

        Returns:
            Image.Image: Combined grid image
        """
        # Calculate grid dimensions
        n = len(images)
        if n <= 1:
            return images[0]

        cols = int(np.ceil(np.sqrt(n)))
        rows = int(np.ceil(n / cols))

        # Get max dimensions
        w_max = max(img.width for img in images)
        h_max = max(img.height for img in images)

        # Create output image
        grid = Image.new("RGB", (w_max * cols, h_max * rows))

        # Paste images into grid
        for idx, img in enumerate(images):
            i = idx // cols
            j = idx % cols
            grid.paste(img, (j * w_max, i * h_max))

        return grid

    def update_image(self, images: Union[Image.Image, list[Image.Image]]) -> None:
        """Update the displayed image(s).

        Args:
            images: Single image or list of images to display
        """
        # Convert single image to list
        if isinstance(images, Image.Image):
            images = [images]

        # Create grid of all images
        grid_img = self.create_image_grid(images)

        # Calculate the aspect ratio of the grid
        aspect_ratio = grid_img.width / grid_img.height

        # Determine the new dimensions while maintaining the aspect ratio
        label_width = int(4 * self.winfo_width() / 7)
        label_height = int(4 * self.winfo_height() / 7)

        if label_width / aspect_ratio <= label_height:
            new_width = label_width
            new_height = int(label_width / aspect_ratio)
        else:
            new_height = label_height
            new_width = int(label_height * aspect_ratio)

        # Resize the grid image
        try:
            grid_img = grid_img.resize((new_width, new_height), Image.LANCZOS)
        except RecursionError:
            pass
        self.img = grid_img
        if self.display_most_recent_image_flag is False:
            self._update_image_label(grid_img)

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
        """Display the most recent image(s) from the output directory."""
        # Get a list of all image files in the output directory
        image_files = glob.glob("./_internal/output/Classic/*")
        image_files += glob.glob("./_internal/output/Adetailer/*")
        image_files += glob.glob("./_internal/output/Flux/*")
        image_files += glob.glob("./_internal/output/HiresFix/*")
        image_files += glob.glob("./_internal/output/Img2Img/*")

        # If there are no image files, return
        if not image_files:
            return

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
            return

        # Display single image or grid of batch
        if len(batch_images) == 1:
            self.update_image(batch_images[0])
        else:
            self.update_image(batch_images)

    def _start_resize_worker(self):
        """Start the resize worker thread"""
        self._resize_thread = threading.Thread(target=self._resize_worker, daemon=True)
        self._resize_thread.start()

    def _resize_worker(self):
        """Worker thread for handling resize operations"""
        while self._resize_running:
            try:
                # Wait for resize event or timeout
                if self._resize_queue.qsize() > 0:
                    event = self._resize_queue.get(timeout=0.1)
                    self._do_resize(event)
                    self._resize_queue.task_done()
                else:
                    self._resize_event.wait(timeout=0.1)
                    self._resize_event.clear()
            except queue.Empty:
                continue

    def _queue_resize(self, event):
        """Queue a resize event"""
        current_time = time.time()

        # Debounce resize events
        if current_time - self._last_resize_time > self._resize_delay:
            self._last_resize_time = current_time
            self._resize_queue.put(event)
            self._resize_event.set()

    def _do_resize(self, event):
        """Handle resize operation in worker thread"""
        width = self.winfo_width()
        height = self.winfo_height()

        # Update UI components in main thread
        self.after(0, lambda: self._update_components(width, height))

        # Update image if exists
        if hasattr(self, "img"):
            self._update_image_threaded(self.img)

    def _update_components(self, width, height):
        """Update UI components sizes"""
        # Update component sizes based on window dimensions
        width = self.winfo_width()
        height = self.winfo_height()

        # Scale text boxes
        prompt_height = int(height * 0.25)
        neg_height = int(height * 0.15)
        self.prompt_entry.configure(height=prompt_height)
        self.neg.configure(height=neg_height)

    def _update_image_threaded(self, img):
        """Thread-safe image update with caching"""
        if img is None:
            return

        with self._resize_lock:
            # Calculate dimensions
            aspect_ratio = img.width / img.height
            label_width = int(4 * self.winfo_width() / 7)
            label_height = int(4 * self.winfo_height() / 7)

            if label_width / aspect_ratio <= label_height:
                new_width = label_width
                new_height = int(label_width / aspect_ratio)
            else:
                new_height = label_height
                new_width = int(label_height * aspect_ratio)

            # Check cache
            cache_key = (new_width, new_height)
            if cache_key in self._image_cache:
                resized_img = self._image_cache[cache_key]
            else:
                try:
                    resized_img = img.resize((new_width, new_height), Image.LANCZOS)
                    self._image_cache[cache_key] = resized_img

                    # Limit cache size
                    if len(self._image_cache) > 5:
                        self._image_cache.pop(next(iter(self._image_cache)))
                except:
                    return
            if not self.display_most_recent_image_flag:
                # Update image in main thread
                self.after(0, lambda: self._update_image_label_safe(resized_img))

    def _update_image_label_safe(self, img):
        """Thread-safe image label update"""
        if not self.display_most_recent_image_flag:
            self._current_image = ImageTk.PhotoImage(img)
            self.image_label.configure(image=self._current_image)

    def _cleanup(self):
        """Clean up threads before closing"""
        self._resize_running = False
        self._resize_event.set()
        if self._resize_thread:
            self._resize_thread.join(timeout=1.0)
        self.destroy()

    def interrupt_generation(self) -> None:
        """Interrupt ongoing image generation process."""
        if not self.is_generating:
            return

        # Set interrupt flag first
        self.interrupt_flag = True

        # Clear CUDA cache and release memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()

        # Stop and cleanup threads
        for thread in self.generation_threads[:]:
            if thread and thread.is_alive():
                thread.join(timeout=1.0)
            if thread in self.generation_threads:
                self.generation_threads.remove(thread)

        # Reset UI state
        self.progress.set(0)
        self.title("LightDiffusion")
        self.generate_button.configure(state="normal")
        self.display_most_recent_image_flag = True

        # Clear any pending resize tasks
        with self._resize_lock:
            self._resize_queue.queue.clear()

        # Reset model state if needed
        if hasattr(self, "checkpointloadersimple_241"):
            del self.checkpointloadersimple_241
            self.ckpt = None

        # Always reset flags
        self.generation_threads.clear()
        # Reset generation state
        self.is_generating = False
        self.generate_button.configure(state="normal")


if __name__ == "__main__":
    from modules.user.app_instance import app

    app.mainloop()
