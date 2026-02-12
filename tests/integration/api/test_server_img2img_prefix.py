import os
import io
import base64
from PIL import Image
from pathlib import Path

import server


def _make_png_data():
    buf = io.BytesIO()
    img = Image.new('RGB', (32, 32), color='green')
    img.save(buf, format='PNG')
    b = buf.getvalue()
    b64 = base64.b64encode(b).decode('ascii')
    data_uri = f"data:image/png;base64,{b64}"
    return data_uri, b


def test_context_request_prefix_mapping():
    from src.Core.Context import Context
    ctx = Context.from_kwargs(prompt='x', request_filename_prefix='LD-REQ-abc123')
    assert ctx.features.request_filename_prefix == 'LD-REQ-abc123'


def test_generate_endpoint_finds_pipeline_written_file(monkeypatch, server_client, tmp_path):
    """Test that /api/generate returns images when pipeline writes an output file with the request prefix."""

    # Fake pipeline that writes a PNG file into output/Img2Img using the provided request_filename_prefix
    def fake_pipeline(*args, request_filename_prefix=None, **kwargs):
        out_dir = Path('output') / 'Img2Img'
        out_dir.mkdir(parents=True, exist_ok=True)
        # Use safe prefix
        prefix = request_filename_prefix or 'LD-REQ-unknown'
        fname = f"{prefix}_LD-I2I_00001_.png"
        path = out_dir / fname
        img = Image.new('RGB', (32, 32), color='purple')
        img.save(path)
        return {}

    monkeypatch.setattr(server, 'pipeline', fake_pipeline)

    data_uri, _ = _make_png_data()
    payload = {
        'prompt': 'test',
        'width': 512,
        'height': 512,
        'num_images': 1,
        'img2img_mode': True,
        'img2img_image': data_uri,
    }

    res = server_client.post('/api/generate', json=payload)
    assert res.status_code == 200, res.text
    j = res.json()
    assert 'image' in j or 'images' in j

    # Clean up any files created in output/Img2Img
    out_dir = Path('output') / 'Img2Img'
    if out_dir.exists():
        for f in out_dir.iterdir():
            try:
                f.unlink()
            except Exception:
                pass
        try:
            out_dir.rmdir()
        except Exception:
            pass