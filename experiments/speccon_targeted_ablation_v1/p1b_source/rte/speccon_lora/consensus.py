"""Cross-block gradient subspace consensus for SpecCon-LoRA."""

import hashlib
import os
import torch


def _seed(module_name, block_index, base_seed):
    payload = f"{module_name}:{block_index}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def randomized_left_svd(grad, rank, oversample=8, power_iterations=1, seed=3407):
    """Frozen randomized SVD, retaining both sides from the same sketch."""
    matrix = grad.detach().float()
    if matrix.ndim > 2:
        matrix = matrix.reshape(matrix.shape[0], -1)
    device = matrix.device
    if not matrix.is_cuda and torch.cuda.is_available():
        device = torch.device("cuda", torch.cuda.current_device())
    matrix = matrix.to(device, non_blocking=True)
    target_rank = min(int(rank), min(matrix.shape))
    sketch_rank = min(target_rank + int(oversample), min(matrix.shape))
    if target_rank <= 0:
        return torch.empty(matrix.shape[0], 0), torch.empty(0), torch.empty(matrix.shape[1], 0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn(matrix.shape[1], sketch_rank, generator=generator).to(device)
    q, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(int(power_iterations)):
        q, _ = torch.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    small = q.T @ matrix
    u_small, singular_values, vh_small = torch.linalg.svd(small, full_matrices=False)
    left = q @ u_small[:, :target_rank]
    right = vh_small[:target_rank, :].T
    return left.cpu(), singular_values[:target_rank].pow(2).cpu(), right.cpu()


def build_block_spectral_profile(block_grad_cache, config, block_index):
    profile = {}
    for name, item in block_grad_cache.items():
        left, energy, right = randomized_left_svd(
            item["grad"],
            rank=config.r_max,
            oversample=config.spectral_oversample,
            power_iterations=config.spectral_power_iterations,
            seed=_seed(name, block_index, config.spectral_seed),
        )
        profile[name] = {"left": left, "right": right, "energy": energy}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profile


def build_consensus_spectral_profiles(block_profiles, config):
    """Compute prefix-subspace agreement invariant to singular-vector signs."""
    side = os.environ.get("SPECCON_SCORE_SIDE", "left")
    if side not in {"left", "right", "joint"}:
        raise ValueError(f"Unsupported score side: {side}")
    result = {}
    module_names = sorted({name for block in block_profiles for name in block})
    for name in module_names:
        items = [block[name] for block in block_profiles if name in block]
        rank = min(item["energy"].numel() for item in items)
        energies = torch.stack([item["energy"][:rank] for item in items])
        consistency = torch.ones(rank)
        if len(items) > 1:
            pairs = len(items) * (len(items) - 1) // 2
            for k in range(1, rank + 1):
                overlap_left = 0.0
                overlap_right = 0.0
                for a in range(len(items)):
                    ua = items[a]["left"][:, :k]
                    va = items[a]["right"][:, :k]
                    for b in range(a + 1, len(items)):
                        ub = items[b]["left"][:, :k]
                        vb = items[b]["right"][:, :k]
                        overlap_left += float((ua.T @ ub).pow(2).sum()) / k
                        if side != "left":
                            overlap_right += float((va.T @ vb).pow(2).sum()) / k
                left_score = overlap_left / pairs
                if side == "left":
                    consistency[k - 1] = left_score
                else:
                    right_score = overlap_right / pairs
                    consistency[k - 1] = right_score if side == "right" else (left_score * right_score) ** 0.5
        result[name] = {
            "mean_energy": energies.mean(0),
            "energy_std": energies.std(0, unbiased=False),
            "consistency": consistency.clamp(config.subspace_consistency_floor, 1.0),
            "num_blocks": len(items),
            "consistency_lambda": float(config.subspace_consistency_lambda),
        }
    return result
