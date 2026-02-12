import pytest
import server


@pytest.mark.asyncio
async def test_telemetry_includes_max_images_per_group():
    t = await server.telemetry()
    assert "max_images_per_group" in t
    assert t["max_images_per_group"] == server.LD_MAX_IMAGES_PER_GROUP
