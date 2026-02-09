import os
import sys
from pathlib import Path

# Add project root to path
project_root = Path(__file__).resolve().parent
sys.path.insert(0, str(project_root))

try:
    from src.FileManaging import Downloader
    print("Initializing Flux2 Klein download...")
    Downloader.CheckAndDownloadFlux2()
    print("\nDownload process finished.")
    print("Models should be located in:")
    print("  - ./include/diffusion_model/ (Diffusion Model)")
    print("  - ./include/text_encoder/ (Text Encoder)")
    print("  - ./include/vae/ (VAE)")
except ImportError as e:
    print(f"Error: Could not import Downloader. Make sure you are running this from the project root. {e}")
except Exception as e:
    print(f"An unexpected error occurred: {e}")