import asyncio

import pytest

pytestmark = pytest.mark.slow


class FakePreviewClient:
    def __init__(self):
        self.payloads = []

    async def send_json(self, payload):
        self.payloads.append(payload)


@pytest.mark.slow
def test_preview_broadcast_payload_contract():
    import server

    client = FakePreviewClient()
    original_clients = list(server._preview_clients)
    server._preview_clients[:] = [client]

    try:
        asyncio.run(
            server.broadcast_preview(
                step=1,
                total_steps=10,
                images=["data:image/png;base64,ZmFrZQ=="],
                message_type="preview",
                generation_id="test-preview",
            )
        )
    finally:
        server._preview_clients[:] = original_clients

    assert len(client.payloads) == 1
    payload = client.payloads[0]
    assert payload["type"] == "preview"
    assert payload["step"] == 1
    assert payload["total_steps"] == 10
    assert payload["generation_id"] == "test-preview"
    assert payload["images"] == ["data:image/png;base64,ZmFrZQ=="]
