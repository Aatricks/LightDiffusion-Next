# LightDiffusion-Next

The purpose of this repository and project is to be refactored version of the original LightDiffusion repository. The goal is to make the code more readable, maintainable, and to add new features to the project. The original repository can be found [here](https://github.com/Aatrick/LightDiffusion).

For more information, check the [documentation](https://aatrick.github.io/LightDiffusion/)

## Features

- Original Txt2Img, Img2Img (R-ERSGAN4x+ UltimateSDUpscaling DPM++ 2M)
- Attention syntax
- Hires-Fix (euler ancestral normal)
- GPU only
- Xformers and Pytorch optimization
- Stable-Fast implementation offering a 70% speedup at the cost of pre inference model optimization windup time
- FP16 and FP32 precision support
- Saved state in between starts
- GUI
- DPM Adaptive Karras
- Clip Skip
- LoRa and textual inversion (embeddings) support
- Automatic Prompt-Enhancing with llama3.1 (ollama)
- Discord bot integration with the installation of [Boubou](https://github.com/Aatrick/Boubou) and the usage of LightDiffusion's pipeline
- Automatic Face and body Detailer (based on [Impact pack](https://github.com/ltdrdata/ComfyUI-Impact-Pack))
- TAESD image previewing at the cost of a 10% performance hit

![Home Image](./HomeImage.png)

## Installation

To install, please download a release or clone this repo and execute the run.bat file in a cmd and you should be good to go.

#### From Source

Else install the python dependencies by writing `pip install -r requirements.txt`

After doing that, add your SD1/1.5 safetensors model to the checkpoints directory and you should be good to go.

### Stable-Fast

To use the stable-fast optimization refer to this [guide](https://github.com/chengzeyi/stable-fast?tab=readme-ov-file#installation) to install from the wheel file.

### Ollama

To use the Prompt enhancer refer to this [guide](https://github.com/ollama/ollama?tab=readme-ov-file) to install and run those commands
`ollama run llama3.1`
`pip install ollama`

### Discord

To use LightDiffusion in discord, refer to the installation guide in this [repo](https://github.com/Aatrick/Boubou)
