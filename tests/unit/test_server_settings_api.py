import importlib
import os

from fastapi.testclient import TestClient


def test_settings_api_endpoints(tmp_path, monkeypatch):
    # Ensure SettingsStore and server use test-local store path
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(tmp_path / "settings_store.json"))

    # Reload modules so they pick up the env var
    # Load SettingsStore directly from file to avoid package-import side effects
    import importlib.util
    spec = importlib.util.spec_from_file_location('settings_store_module', os.path.join(os.getcwd(), 'src', 'Core', 'SettingsStore.py'))
    SS = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(SS)

    import server
    importlib.reload(server)

    client = TestClient(server.app)

    # Initially no seed
    r = client.get("/api/settings/last")
    assert r.status_code == 200
    assert r.json().get("seed") is None

    # Set seed via SettingsStore and read through API
    SS.set_last_seed(123456)
    r = client.get("/api/settings/last")
    assert r.status_code == 200
    assert r.json().get("seed") == 123456

    # Post a settings snapshot
    payload = {
        "settings": {
            "prompt": "unit test",
            "width": 64,
            "height": 64,
            "num_images": 1,
            "batch_size": 1,
            "steps": 10,
            "cfg_scale": 7.0,
            "seed": 42,
            "scheduler": "ays",
            "sampler": "dpmpp_sde_cfgpp",
            "model_path": "test-model",
            "img2img_mode": False,
            "img2img_denoise": 0.75,
            "hiresfix": False,
            "adetailer": False,
            "enhance_prompt": False,
            "stable_fast": False,
            "reuse_seed": False,
            "keep_models_loaded": True,
            "enable_preview": False
        }
    }
    r = client.post("/api/settings/history", json=payload)
    assert r.status_code == 200
    snap = r.json().get("snapshot")
    assert snap and snap.get("id") and snap.get("ts")

    # GET history
    r = client.get("/api/settings/history")
    assert r.status_code == 200
    hist = r.json().get("history")
    assert isinstance(hist, list) and len(hist) >= 1
    entry = hist[0]["settings"]
    # prompt should be removed by server-side sanitization
    assert "prompt" not in entry
    # verify allowed parameters are preserved
    assert entry["steps"] == 10
    assert entry["cfg_scale"] == 7.0
    assert entry["sampler"] == "dpmpp_sde_cfgpp"
    assert entry["scheduler"] == "ays"
    assert entry["model_path"] == "test-model"
    assert entry["width"] == 64
    assert entry["height"] == 64
    assert entry["seed"] == 42

    # Now explicitly store prompt via include_prompt flag
    r = client.post("/api/settings/history", json={"settings": payload["settings"], "include_prompt": True})
    assert r.status_code == 200
    snap2 = r.json().get("snapshot")
    assert snap2 and snap2.get("id")

    r = client.get("/api/settings/history")
    assert r.status_code == 200
    hist2 = r.json().get("history")
    # Newest-first; explicit include_prompt entry should be first and contain the prompt
    assert isinstance(hist2, list) and len(hist2) >= 1
    assert hist2[0]["settings"].get("prompt") == "unit test"
