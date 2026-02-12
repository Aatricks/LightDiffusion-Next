import os
import base64
import io
import tempfile

from PIL import Image

import server


def _make_png_data():
    buf = io.BytesIO()
    img = Image.new('RGB', (32, 32), color='red')
    img.save(buf, format='PNG')
    b = buf.getvalue()
    b64 = base64.b64encode(b).decode('ascii')
    data_uri = f"data:image/png;base64,{b64}"
    return data_uri, b


def test_save_data_uri():
    data_uri, raw = _make_png_data()
    path = server._save_img2img_image_to_file(data_uri, max_size_bytes=10 * 1024 * 1024)
    assert path and os.path.exists(path)
    with open(path, 'rb') as f:
        content = f.read()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    os.remove(path)


def test_save_bare_base64():
    data_uri, raw = _make_png_data()
    b64 = data_uri.split(',', 1)[1]
    path = server._save_img2img_image_to_file(b64, max_size_bytes=10 * 1024 * 1024)
    assert path and os.path.exists(path)
    with open(path, 'rb') as f:
        content = f.read()
    assert content.startswith(b"\x89PNG\r\n\x1a\n")
    os.remove(path)


def test_generate_endpoint_converts(monkeypatch, server_client):
    data_uri, raw = _make_png_data()

    async def fake_enqueue(pending):
        # The pending request should have had its img2img_image converted to a real file path
        val = pending.req.img2img_image
        assert val and os.path.exists(val)
        with open(val, 'rb') as f:
            d = f.read()
        assert d.startswith(b"\x89PNG\r\n\x1a\n")
        return {'image': 'data:image/png;base64,' + base64.b64encode(raw).decode('ascii')}

    monkeypatch.setattr(server._generation_buffer, 'enqueue', fake_enqueue)

    payload = {
        'prompt': 'test',
        'width': 512,
        'height': 512,
        'num_images': 1,
        'img2img_mode': True,
        'img2img_image': data_uri,
    }

    res = server_client.post('/api/generate', json=payload)
    assert res.status_code == 200
    assert 'image' in res.json()


def test_save_too_large():
    # Create a base64 string that decodes to more than 10MB
    big_bytes = b'A' * (11 * 1024 * 1024)
    big_b64 = base64.b64encode(big_bytes).decode('ascii')
    from fastapi import HTTPException

    try:
        server._save_img2img_image_to_file(big_b64, max_size_bytes=10 * 1024 * 1024)
        raised = False
    except Exception as e:
        raised = True
        assert isinstance(e, HTTPException)
        assert e.status_code == 413

    assert raised
