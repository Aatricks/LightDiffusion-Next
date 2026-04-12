import asyncio
import contextlib

import pytest

import server


@pytest.mark.asyncio
async def test_img2img_requests_with_different_images_do_not_coalesce(monkeypatch):
    buf = server.GenerationBuffer()
    calls: list[list[str]] = []

    async def fake_process_group(items):
        calls.append([item.request_id for item in items])
        for item in items:
            item.future.set_result({"image": f"data:image/png;base64,{item.request_id}"})

    async def noop_prefetch(_signature):
        return None

    monkeypatch.setattr(buf, "_process_group", fake_process_group)
    monkeypatch.setattr(buf, "_look_ahead_and_prefetch", noop_prefetch)

    req1 = server.GenerateRequest(
        prompt="p1",
        img2img_mode=True,
        img2img_image="/tmp/input-a.png",
    )
    req2 = server.GenerateRequest(
        prompt="p2",
        img2img_mode=True,
        img2img_image="/tmp/input-b.png",
    )

    pending1 = server.PendingRequest(req1, request_id="r1")
    pending2 = server.PendingRequest(req2, request_id="r2")

    async with buf._lock:
        buf._pending.extend([pending1, pending2])
        buf._new_request.set()

    worker = asyncio.create_task(buf._worker())
    try:
        await asyncio.wait_for(asyncio.gather(pending1.future, pending2.future), timeout=1.0)
    finally:
        worker.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await worker

    assert calls == [["r1"], ["r2"]]
