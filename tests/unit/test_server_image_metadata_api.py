import base64
import io
import importlib

from fastapi.testclient import TestClient
from PIL import Image, PngImagePlugin


def test_image_metadata_endpoint(tmp_path, monkeypatch):
    # Ensure server loads with test environment
    monkeypatch.setenv("LD_SETTINGS_STORE_PATH", str(tmp_path / "settings_store.json"))

    import server
    importlib.reload(server)
    client = TestClient(server.app)

    # Create an in-memory PNG with PngInfo metadata
    img = Image.new("RGB", (64, 64), color=(255, 0, 0))
    pnginfo = PngImagePlugin.PngInfo()
    pnginfo.add_text("prompt", "import-me")
    pnginfo.add_text("negative_prompt", "bad, ugly")
    pnginfo.add_text("seed", "424242")
    pnginfo.add_text("steps", "12")
    pnginfo.add_text("cfg", "6.5")
    pnginfo.add_text("sampler", "dpmpp_sde_cfgpp")
    pnginfo.add_text("scheduler", "ays")
    pnginfo.add_text("model_path", "test-model")
    pnginfo.add_text("width", "512")
    pnginfo.add_text("height", "512")

    buf = io.BytesIO()
    img.save(buf, format="PNG", pnginfo=pnginfo)
    b64 = base64.b64encode(buf.getvalue()).decode("utf-8")

    # POST data URL
    r = client.post("/api/images/metadata", json={"image": f"data:image/png;base64,{b64}"})
    assert r.status_code == 200
    md = r.json().get("metadata")
    assert md is not None
    assert md.get("prompt") == "import-me"
    assert md.get("negative_prompt") == "bad, ugly"
    assert md.get("seed") == 424242
    assert md.get("steps") == 12
    assert abs(md.get("cfg_scale") - 6.5) < 1e-6
    assert md.get("sampler") == "dpmpp_sde_cfgpp"
    assert md.get("scheduler") == "ays"
    assert md.get("model_path") == "test-model"
    assert md.get("width") == 512
    assert md.get("height") == 512
