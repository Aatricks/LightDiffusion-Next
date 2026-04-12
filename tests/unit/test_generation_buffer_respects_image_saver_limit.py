import asyncio

import pytest

import server
from src.FileManaging import ImageSaver


@pytest.mark.asyncio
async def test_chunking_respects_image_saver_limit(monkeypatch):
    server.LD_MAX_IMAGES_PER_GROUP = 32

    # Force a small ImageSaver limit so the buffer must chunk accordingly
    monkeypatch.setattr(ImageSaver, "MAX_IMAGES_PER_SAVE", 3)

    calls = []

    def fake_pipeline(**kwargs):
        # Record invocation and the number of per-sample entries
        calls.append(kwargs)
        per_sample_info = kwargs.get("per_sample_info", [])
        results = {}
        for info in per_sample_info:
            rid = info["request_id"]
            filename = f"{rid}_{len(results.get(rid, []))}_img.png"
            ImageSaver.store_image_bytes(info["filename_prefix"], filename, "Classic", b"PNGDATA")
            results.setdefault(rid, []).append({"filename": filename, "subfolder": "Classic"})
        return {"batched_results": results}

    monkeypatch.setattr(server, "pipeline", fake_pipeline)

    # Create 10 small requests (1 image each) with a large batch_size so the
    # saver limit, not the request batch size, is what forces chunking.
    items = [
        server.PendingRequest(
            server.GenerateRequest(prompt=f"p{i}", num_images=1, batch_size=10),
            request_id=str(i),
        )
        for i in range(10)
    ]

    buf = server.GenerationBuffer()

    await buf._process_group(items)

    # Expect chunking to force multiple pipeline invocations
    assert len(calls) == 4
    # Each chunk should have at most 3 images as per ImageSaver.MAX_IMAGES_PER_SAVE
    for c in calls:
        assert c["number"] <= 3

    # Ensure all requests were completed
    for p in items:
        assert p.future.done()
        res = p.future.result()
        assert isinstance(res, dict)
        assert "images" in res or "image" in res
