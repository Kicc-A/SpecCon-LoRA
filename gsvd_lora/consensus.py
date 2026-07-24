"""Cross-block gradient subspace consensus for GSVD-LoRA."""

import hashlib
import torch


def _seed(module_name, block_index, base_seed):
    payload = f"{module_name}:{block_index}:{base_seed}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "little") % (2**31)


def randomized_left_svd(grad, rank, oversample=8, power_iterations=1, seed=3407):
    """Deterministic randomized SVD retaining only the leading left subspace."""
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
        return torch.empty(matrix.shape[0], 0), torch.empty(0)

    generator = torch.Generator(device="cpu")
    generator.manual_seed(int(seed))
    omega = torch.randn(matrix.shape[1], sketch_rank, generator=generator).to(device)
    q, _ = torch.linalg.qr(matrix @ omega, mode="reduced")
    for _ in range(int(power_iterations)):
        q, _ = torch.linalg.qr(matrix @ (matrix.T @ q), mode="reduced")
    small = q.T @ matrix
    u_small, singular_values, _ = torch.linalg.svd(small, full_matrices=False)
    left = q @ u_small[:, :target_rank]
    return left.cpu(), singular_values[:target_rank].pow(2).cpu()


def build_block_spectral_profile(block_grad_cache, config, block_index):
    profile = {}
    for name, item in block_grad_cache.items():
        left, energy = randomized_left_svd(
            item["grad"],
            rank=config.r_max,
            oversample=config.spectral_oversample,
            power_iterations=config.spectral_power_iterations,
            seed=_seed(name, block_index, config.spectral_seed),
        )
        profile[name] = {"left": left, "energy": energy}
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return profile


def build_consensus_spectral_profiles(block_profiles, config):
    """Compute prefix-subspace agreement invariant to singular-vector signs."""
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
                overlap = 0.0
                for a in range(len(items)):
                    ua = items[a]["left"][:, :k]
                    for b in range(a + 1, len(items)):
                        ub = items[b]["left"][:, :k]
                        overlap += float((ua.T @ ub).pow(2).sum()) / k
                consistency[k - 1] = overlap / pairs
        result[name] = {
            "mean_energy": energies.mean(0),
            "energy_std": energies.std(0, unbiased=False),
            "consistency": consistency.clamp(config.subspace_consistency_floor, 1.0),
            "num_blocks": len(items),
            "consistency_lambda": float(config.subspace_consistency_lambda),
        }
    return result
