from __future__ import annotations

from collections.abc import Sequence
from enum import Enum

import numpy as np


class DState(str, Enum):
    STABLE = "STABLE"
    STRESSED = "STRESSED"
    DISRUPTED = "DISRUPTED"
    CHAOTIC = "CHAOTIC"
    RECOVERING = "RECOVERING"
    RUINED = "RUINED"


def local_chaos_pressure(
    disruption_intensity: float, congestion_memory: float, local_failed_qdot_ratio: float
) -> float:
    return max(0.0, disruption_intensity + congestion_memory + local_failed_qdot_ratio)


def global_chaos_pressure(
    local_pressures: Sequence[float] | np.ndarray, late_ratio: float
) -> float:
    if len(local_pressures) == 0:
        return late_ratio
    return float(np.mean(local_pressures)) + late_ratio


def local_order_pressure(active_qdots: int, moving_qdots: int, failed_qdots: int) -> float:
    return max(0.0, float(active_qdots + moving_qdots - failed_qdots))


def global_order_pressure(active_qdots: int, moving_qdots: int, late_qdots: int) -> float:
    """Share of active Q dots that are flowing, minus the share that is late."""
    active = max(active_qdots, 1)
    return moving_qdots / active - late_qdots / active


def choose_d_state(
    system_ruined: bool,
    current_chaos: float,
    previous_chaos: float,
    stressed_threshold: float,
    disrupted_threshold: float,
    chaotic_threshold: float,
    recovering_decay_ratio: float,
) -> DState:
    """Pick the grid regime.

    Thresholds on chaos pressure decide CHAOTIC and DISRUPTED outright. Below
    the disrupted threshold, the grid is RECOVERING when pressure was at
    least stressed on the previous step and has since dropped by more than
    `recovering_decay_ratio`.
    """
    if system_ruined:
        return DState.RUINED
    if current_chaos >= chaotic_threshold:
        return DState.CHAOTIC
    if current_chaos >= disrupted_threshold:
        return DState.DISRUPTED
    if previous_chaos >= stressed_threshold and current_chaos < previous_chaos * (
        1.0 - recovering_decay_ratio
    ):
        return DState.RECOVERING
    if current_chaos >= stressed_threshold:
        return DState.STRESSED
    return DState.STABLE
