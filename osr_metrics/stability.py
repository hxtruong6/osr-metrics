from __future__ import annotations


def stability_score(prob_trajectory: list[float]) -> float:
    """Compute Stability Score for a generation trajectory.

    Stab(ℓ_k) = 1 - (1/T) * Σ_t |P(y_t = v_yes | ·) - p̄|

    where p̄ = mean probability over T steps.
    Higher = more stable (consistent confidence, less spiky).

    Args:
        prob_trajectory: list of T yes-token probabilities, one per step.

    Returns:
        Stability score in [0, 1].
    """
    if len(prob_trajectory) == 0:
        return 0.0
    T = len(prob_trajectory)
    mean_p = sum(prob_trajectory) / T
    mad = sum(abs(p - mean_p) for p in prob_trajectory) / T
    return 1.0 - mad
