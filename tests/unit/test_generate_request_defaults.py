from server import GenerateRequest


def test_generate_request_preview_fidelity_default():
    req = GenerateRequest(prompt="test")
    assert req.preview_fidelity == "balanced"
