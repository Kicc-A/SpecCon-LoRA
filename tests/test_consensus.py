import torch

from speccon_lora.config import SpecConLoraConfig
from speccon_lora.consensus import build_consensus_spectral_profiles


def _config():
    return SpecConLoraConfig(
        target_modules=["q_proj"],
        r_max=2,
        subspace_consistency_floor=0.0,
        subspace_consistency_lambda=1.0,
    )


def test_identical_subspaces_have_unit_consistency():
    u = torch.eye(4)[:, :2]
    profiles = [
        {"m": {"left": u, "energy": torch.tensor([4.0, 1.0])}},
        {"m": {"left": -u, "energy": torch.tensor([4.0, 1.0])}},
    ]
    result = build_consensus_spectral_profiles(profiles, _config())["m"]
    assert torch.allclose(result["consistency"], torch.ones(2), atol=1e-6)


def test_orthogonal_leading_subspaces_have_zero_consistency():
    profiles = [
        {"m": {"left": torch.eye(4)[:, :1], "energy": torch.tensor([1.0])}},
        {"m": {"left": torch.eye(4)[:, 1:2], "energy": torch.tensor([1.0])}},
    ]
    result = build_consensus_spectral_profiles(profiles, _config())["m"]
    assert result["consistency"].item() < 1e-6
