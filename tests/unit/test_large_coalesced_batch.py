import pytest
import server
from src.FileManaging import ImageSaver


@pytest.mark.asyncio
async def test_large_coalesced_batch_1024(monkeypatch):
    # Set small chunk limit to force many chunks
    server.LD_MAX_IMAGES_PER_GROUP = 32

    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", immediate_to_thread)

    def fake_pipeline(**kwargs):
        per_sample_info = kwargs.get("per_sample_info", [])
        results = {}
        for info in per_sample_info:
            rid = info["request_id"]
            filename = f"{rid}_{len(results.get(rid, []))}_img.png"
            ImageSaver.store_image_bytes(f"LD-REQ-{rid}", filename, "Classic", b"PNGDATA")
            results.setdefault(rid, []).append({"filename": filename, "subfolder": "Classic"})
        return {"batched_results": results}

    monkeypatch.setattr(server, "pipeline", fake_pipeline)

    items = []
    # 64 requests, each asking for 16 images = 1024 total
    for i in range(64):
        req = server.GenerateRequest(prompt=f"p{i}", num_images=16)
        pr = server.PendingRequest(req, request_id=f"r{i:04d}")
        items.append(pr)

    buf = server.GenerationBuffer()

    await buf._process_group(items)

    # All completed
    for p in items:
        assert p.future.done()
        res = p.future.result()
        assert isinstance(res, dict)
        assert "images" in res and len(res["images"]) == 16

    # Buffer emptied for all
    for i in range(64):
        assert ImageSaver.pop_image_bytes(f"LD-REQ-r{i:04d}") == []
