import json
import time
import pytest

pytestmark = pytest.mark.slow


@pytest.mark.slow
def test_preview_with_testclient(server_client):
    """Connect to the preview websocket via TestClient and request a preview."""
    # Start websocket connection
    with server_client.websocket_connect('/ws/preview') as websocket:
        # Fire a generate request that enables preview
        payload = {
            "prompt": "a beautiful landscape",
            "width": 512,
            "height": 512,
            "steps": 10,
            "enable_preview": True,
        }

        resp = server_client.post('/api/generate', json=payload)
        assert resp.status_code == 200

        previews_received = 0
        start_time = time.time()
        while time.time() - start_time < 30:
            try:
                message = websocket.receive_text(timeout=5.0)
            except Exception:
                break
            data = json.loads(message)
            if data.get('type') == 'preview':
                previews_received += 1
                if previews_received == 1 and 'images' in data and data['images']:
                    assert isinstance(data['images'][0], str)

        assert previews_received > 0, "Expected at least one preview message via websocket"
