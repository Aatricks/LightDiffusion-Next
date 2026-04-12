import asyncio
import base64

import pytest

import server
from src.FileManaging import ImageSaver


@pytest.mark.asyncio
async def test_single_request_large_num_images_is_chunked(monkeypatch):
    server.LD_MAX_IMAGES_PER_GROUP = 3

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

    req = server.GenerateRequest(prompt="p", num_images=10)
    pr = server.PendingRequest(req, request_id="r_big")
    buf = server.GenerationBuffer()

    await buf._process_group([pr])

    assert pr.future.done()
    res = pr.future.result()
    # Should have produced multiple images
    assert isinstance(res, dict)
    if "images" in res:
        assert len(res["images"]) == 10
    else:
        # Single image case should not happen for num_images=10
        pytest.fail("Expected 10 images in response")

    # Buffer should be emptied
    assert ImageSaver.pop_image_bytes("LD-REQ-r_big") == []


@pytest.mark.asyncio
async def test_single_request_respects_batch_size_semantics(monkeypatch):
    server.LD_MAX_IMAGES_PER_GROUP = 32
    monkeypatch.setattr(ImageSaver, "MAX_IMAGES_PER_SAVE", 16)

    calls = []

    def fake_pipeline(**kwargs):
        calls.append((kwargs["number"], kwargs["batch"]))
        per_sample_info = kwargs.get("per_sample_info", [])
        results = {}
        for idx, info in enumerate(per_sample_info):
            rid = info["request_id"]
            filename = f"{rid}_{idx}_img.png"
            ImageSaver.store_image_bytes(info["filename_prefix"], filename, "Classic", b"PNGDATA")
            results.setdefault(rid, []).append({"filename": filename, "subfolder": "Classic"})
        return {"batched_results": results}

    monkeypatch.setattr(server, "pipeline", fake_pipeline)

    req = server.GenerateRequest(prompt="p", num_images=5, batch_size=2)
    pr = server.PendingRequest(req, request_id="r_batch")
    buf = server.GenerationBuffer()

    await buf._process_group([pr])

    assert calls == [(2, 2), (2, 2), (1, 1)]
    assert pr.future.done()
    res = pr.future.result()
    assert isinstance(res, dict)
    assert "images" in res
    assert len(res["images"]) == 5
    assert ImageSaver.pop_image_bytes("LD-REQ-r_batch") == []
