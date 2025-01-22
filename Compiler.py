import os
import re

from httpx import get

# list every file in the modules directory



files_ordered = [
    "./modules/Utilities/util.py",
    "./modules/sample/sampling_util.py",
    "./modules/Device/Device.py",
    "./modules/cond/cond_util.py",
    "./modules/cond/cond.py",
    "./modules/sample/ksampler_util.py",
    "./modules/cond/cast.py",
    "./modules/Attention/AttentionMethods.py",
    "./modules/AutoEncoders/taesd.py",
    "./modules/cond/cond.py",
    "./modules/cond/Activation.py",
    "./modules/Attention/Attention.py",
    "./modules/sample/samplers.py",
    "./modules/sample/CFG.py",
    "./modules/NeuralNetwork/transformer.py",
    "./modules/sample/sampling.py",
    "./modules/clip/CLIPTextModel.py",
    "./modules/AutoEncoders/ResBlock.py",
    "./modules/AutoDetailer/mask_util.py",
    "./modules/NeuralNetwork/unet.py",
    "./modules/SD15/SDClip.py",
    "./modules/SD15/SDToken.py",
    "./modules/UltimateSDUpscale/USDU_util.py",
    "./modules/StableFast/SF_util.py",
    "./modules/Utilities/Latent.py",
    "./modules/AutoDetailer/SEGS.py",
    "./modules/AutoDetailer/tensor_util.py",
    "./modules/AutoDetailer/AD_util.py",
    "./modules/clip/FluxClip.py",
    "./modules/Model/ModelPatcher.py",
    "./modules/Model/ModelBase.py",
    "./modules/UltimateSDUpscale/image_util.py",
    "./modules/UltimateSDUpscale/RDRB.py",
    "./modules/StableFast/ModuleFactory.py",
    "./modules/AutoDetailer/bbox.py",
    "./modules/AutoEncoders/VariationalAE.py",
    "./modules/clip/Clip.py",
    "./modules/Model/LoRas.py",
    "./modules/BlackForest/Flux.py",
    "./modules/UltimateSDUpscale/USDU_upscaler.py",
    "./modules/StableFast/ModuleTracing.py",
    "./modules/hidiffusion/utils.py",
    "./modules/FileManaging/Downloader.py",
    "./modules/AutoDetailer/SAM.py",
    "./modules/AutoDetailer/ADetailer.py",
    "./modules/Quantize/Quantizer.py",
    "./modules/FileManaging/Loader.py",
    "./modules/SD15/SD15.py",
    "./modules/UltimateSDUpscale/UltimateSDUpscale.py",
    "./modules/StableFast/StableFast.py",
    "./modules/hidiffusion/msw_msa_attention.py",
    "./modules/FileManaging/ImageSaver.py",
    "./modules/Utilities/Enhancer.py",
    "./modules/Utilities/upscale.py",
    "./modules/user/pipeline.py",
]

def get_file_patterns():
    patterns = []
    seen = set()
    for path in files_ordered:
        filename = os.path.basename(path)
        name = os.path.splitext(filename)[0]
        if name not in seen:
            # Pattern 1: matches module name when not in brackets or after a dot
            pattern1 = rf'(?<![a-zA-Z0-9_\.])({name}\.)(?![)\]])'
            # Pattern 2: matches module name inside brackets while preserving them  
            pattern2 = rf'(\[|\()({name}\.)([^\]\)]+?)(\]|\))'
            pattern3 = 'cond_util\.'
            patterns.extend([
                (pattern1, ''),  # Remove module name and dot outside brackets
                (pattern2, r'\1\3\4'),  # Keep brackets, remove only module name
                (pattern3, '')
            ])
            seen.add(name)
    return patterns

def remove_file_names(line):
    patterns = get_file_patterns()
    result = line
    for pattern, replacement in patterns:
        result = re.sub(pattern, replacement, result)
    return result

try:
    with open("./compiled.py", "w") as output_file:
        for file_path in files_ordered:
            try:
                with open(file_path, "r") as input_file:
                    for line in input_file:
                        if not line.lstrip().startswith("from modules."):
                            # Apply the file name removal before writing
                            modified_line = remove_file_names(line)
                            output_file.write(modified_line)
                    output_file.write("\n\n")
                print(f"Processed: {file_path}")
            except FileNotFoundError:
                print(f"Error: Could not find file {file_path}")
            except Exception as e:
                print(f"Error processing {file_path}: {str(e)}")
except Exception as e:
    print(f"Error creating compiled.py: {str(e)}")