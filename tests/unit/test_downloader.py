import pytest
from pathlib import Path
from src.FileManaging import Downloader


def test_check_and_download_flux2_downloads_when_only_tokenizer_present(tmp_path, monkeypatch):
    cwd = tmp_path
    inc = cwd / "include"
    # create only a tokenizer directory in text_encoder
    (inc / "text_encoder" / "qwen25_tokenizer").mkdir(parents=True, exist_ok=True)
    # create diffusion and vae files so only text encoder needs download
    (inc / "diffusion_model").mkdir(parents=True, exist_ok=True)
    (inc / "diffusion_model" / "flux-2-klein-4b.safetensors").write_text("dummy")
    (inc / "vae").mkdir(parents=True, exist_ok=True)
    (inc / "vae" / "flux2-vae.safetensors").write_text("dummy")

    monkeypatch.chdir(cwd)

    calls = []

    def fake_hf_hub_download(repo_id=None, filename=None, local_dir=None, **kwargs):
        calls.append((repo_id, filename, local_dir))
        # simulate a nested download (subfolder) to test move-up logic
        dest_subdir = Path(local_dir) / "split_files"
        dest_subdir.mkdir(parents=True, exist_ok=True)
        fpath = dest_subdir / Path(filename).name
        fpath.write_text("dummy")
        return str(fpath)

    monkeypatch.setattr("src.FileManaging.Downloader.hf_hub_download", fake_hf_hub_download)

    Downloader.CheckAndDownloadFlux2()

    # assert that the Qwen encoder was requested
    assert any("qwen_3_4b.safetensors" in call[1] for call in calls)

    # assert the file is now present at top-level text_encoder dir
    assert (inc / "text_encoder" / "qwen_3_4b.safetensors").exists()
    # and the nested subfolder was cleaned up
    assert not (inc / "text_encoder" / "split_files").exists()


def test_check_and_download_flux2_skips_if_weights_present(tmp_path, monkeypatch):
    cwd = tmp_path
    inc = cwd / "include"
    (inc / "text_encoder").mkdir(parents=True, exist_ok=True)
    # create the actual encoder weights file to simulate presence
    (inc / "text_encoder" / "qwen_3_4b.safetensors").write_text("dummy")
    (inc / "diffusion_model").mkdir(parents=True, exist_ok=True)
    (inc / "diffusion_model" / "flux-2-klein-4b.safetensors").write_text("dummy")
    (inc / "vae").mkdir(parents=True, exist_ok=True)
    (inc / "vae" / "flux2-vae.safetensors").write_text("dummy")

    monkeypatch.chdir(cwd)

    def fake_hf_hub_download(*args, **kwargs):
        raise AssertionError("hf_hub_download should not be called when weights are present")

    monkeypatch.setattr("src.FileManaging.Downloader.hf_hub_download", fake_hf_hub_download)

    # Should not raise
    Downloader.CheckAndDownloadFlux2()

    # weights remain
    assert (inc / "text_encoder" / "qwen_3_4b.safetensors").exists()
