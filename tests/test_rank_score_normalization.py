import pytest

from gsvd_lora.rank_allocator import _normalize_candidate_scores


def test_target_module_normalization_equalizes_group_means():
    candidates = [
        (100.0, 1.0, 10, "model.layers.0.self_attn.q_proj", 4),
        (200.0, 1.0, 10, "model.layers.1.self_attn.q_proj", 4),
        (1.0, 1.0, 10, "model.layers.0.self_attn.v_proj", 4),
        (3.0, 1.0, 10, "model.layers.1.self_attn.v_proj", 4),
    ]

    normalized = _normalize_candidate_scores(candidates, "target_module")
    q_scores = [row[0] for row in normalized if row[3].endswith("q_proj")]
    v_scores = [row[0] for row in normalized if row[3].endswith("v_proj")]

    assert sum(q_scores) / len(q_scores) == pytest.approx(1.0)
    assert sum(v_scores) / len(v_scores) == pytest.approx(1.0)
    assert q_scores[1] / q_scores[0] == pytest.approx(2.0)
    assert v_scores[1] / v_scores[0] == pytest.approx(3.0)


def test_none_preserves_candidate_scores():
    candidates = [(7.0, 2.0, 10, "model.layers.0.self_attn.q_proj", 4)]
    assert _normalize_candidate_scores(candidates, "none") == candidates


def test_unknown_normalization_is_rejected():
    with pytest.raises(ValueError, match="Unsupported rank_score_normalization"):
        _normalize_candidate_scores([], "unknown")
