import server
import pytest


@pytest.mark.asyncio
async def test_generate_endpoint_forwards_preview_fidelity(monkeypatch, async_server_client):
    captured = {}

    async def fake_enqueue(pending):
        # the pending request should have preview_fidelity forwarded
        captured['preview_fidelity'] = pending.req.preview_fidelity
        return {'image': 'data:image/png;base64,xyz'}

    monkeypatch.setattr(server._generation_buffer, 'enqueue', fake_enqueue)

    payload = {
        'prompt': 'test',
        'width': 512,
        'height': 512,
        'num_images': 1,
        'preview_fidelity': 'high',
        'enable_preview': True,
    }

    res = await async_server_client.post('/api/generate', json=payload)
    assert res.status_code == 200
    assert captured.get('preview_fidelity') == 'high'
