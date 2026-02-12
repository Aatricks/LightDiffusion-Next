import torch
import pytest

from src.NeuralNetwork.flux2.layers import attention


def test_attention_raises_on_rope_seq_length_mismatch():
    # Create q/k/v with seq length 12
    q = torch.randn(1, 2, 12, 64)
    k = torch.randn_like(q)
    v = torch.randn_like(q)

    # Create pe with mismatched seq length (13)
    pe = torch.randn(1, 1, 13, 64 // 2, 2, 2)

    with pytest.raises(ValueError) as exc:
        attention(q, k, v, pe)

    assert "RoPE sequence length mismatch" in str(exc.value)
