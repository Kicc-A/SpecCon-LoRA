from dataclasses import dataclass
from typing import List, Optional, Union


@dataclass
class SpecConLoraConfig:
    target_modules: List[str]
    target_prefix: Optional[str] = None
    base_rank: int = 8
    tau: float = 0.90
    r_min: int = 2
    r_max: int = 16
    lora_alpha: int = 16
    lora_dropout: float = 0.05
    calibration_steps: int = 16
    calibration_blocks: int = 4
    use_subspace_consistency: bool = True
    subspace_consistency_lambda: float = 1.0
    subspace_consistency_floor: float = 0.05
    spectral_oversample: int = 8
    spectral_power_iterations: int = 1
    spectral_seed: int = 3407
    adaptive_rank: bool = True
    bias: str = "none"
    init_method: str = "none"
    init_scale: Union[float, str] = 1e-3
    init_a_rms: float = 0.0
    init_auto_scale_min: float = 0.3
    init_auto_scale_max: float = 1.2
    init_energy_beta: float = 0.5
    init_energy_eps: float = 1e-8
    init_small_b_scale: float = 1e-4
    scaling_mode: str = "rank"
    rank_budget_mode: str = "independent"
    rank_budget_grouping: str = "none"
    param_budget: Optional[int] = None
    strict_param_budget: bool = False
    rank_score_gamma: Union[float, str] = 1.0
    rank_score_normalization: str = "none"
    attention_rank_prior: Union[float, str] = 1.0
