import asyncio
import base64
import os

import pytest

import server
from src.FileManaging import ImageSaver


@pytest.mark.asyncio
async def test_process_group_chunks_large_batch(monkeypatch):
    """When many small requests are coalesced, the buffer should split them
    into smaller pipeline runs and merge results correctly."""

    # Keep chunk size small for the test
    server.LD_MAX_IMAGES_PER_GROUP = 3

    async def immediate_to_thread(func, /, *args, **kwargs):
        return func(*args, **kwargs)

    monkeypatch.setattr(server.asyncio, "to_thread", immediate_to_thread)

    # Prepare a simple fake pipeline that records saved images in the
    # in-memory byte buffer and returns a batched_results mapping.
    def fake_pipeline(**kwargs):
        per_sample_info = kwargs.get("per_sample_info", [])
        assert kwargs["batch"] <= 3
        results = {}
        for info in per_sample_info:
            rid = info["request_id"]
            filename = f"{rid}_img.png"
            # Simulate in-memory bytes being stored by SaveImage
            ImageSaver.store_image_bytes(f"LD-REQ-{rid}", filename, "Classic", b"PNGDATA")
            results.setdefault(rid, []).append({"filename": filename, "subfolder": "Classic"})
        return {"batched_results": results}

    monkeypatch.setattr(server, "pipeline", fake_pipeline)

    # Build a list of 7 pending requests which will be chunked as 3,3,1
    items = []
    for i in range(7):
        req = server.GenerateRequest(prompt=f"p{i}", num_images=1, batch_size=3)
        pr = server.PendingRequest(req, request_id=f"r{i:03d}")
        items.append(pr)

    buf = server.GenerationBuffer()

    # Call the internal _process_group directly
    await buf._process_group(items)

    # Ensure all futures were completed successfully and have images
    for p in items:
        assert p.future.done()
        res = p.future.result()
        # Each should resolve to a single image dict
        assert isinstance(res, dict)
        assert ("image" in res) or ("images" in res)

    # Verify that the in-memory buffer is empty for these prefixes
    for i in range(7):
        assert ImageSaver.pop_image_bytes(f"LD-REQ-r{i:03d}") == []
