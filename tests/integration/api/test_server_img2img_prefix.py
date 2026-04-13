import os
import io
import base64
from PIL import Image
from pathlib import Path

import pytest
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


@pytest.mark.asyncio
async def test_generate_endpoint_accepts_img2img_request(monkeypatch, async_server_client):
    """Test that /api/generate accepts img2img payloads and returns an image response."""

    async def fake_enqueue(pending):
        assert pending.req.img2img_mode is True
        assert pending.req.img2img_image
        return {'image': 'data:image/png;base64,xyz'}

    monkeypatch.setattr(server._generation_buffer, 'enqueue', fake_enqueue)

    data_uri, _ = _make_png_data()
    payload = {
        'prompt': 'test',
        'width': 512,
        'height': 512,
        'num_images': 1,
        'img2img_mode': True,
        'img2img_image': data_uri,
    }

    res = await async_server_client.post('/api/generate', json=payload)
    assert res.status_code == 200, res.text
    j = res.json()
    assert j.get('image') == 'data:image/png;base64,xyz'
