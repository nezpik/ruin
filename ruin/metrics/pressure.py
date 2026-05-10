from __future__ import annotations

from enum import Enum
from statistics import mean


class DState(str, Enum):
    STABLE = "STABLE"
    STRESSED = "STRESSED"
    DISRUPTED = "DISRUPTED"
    CHAOTIC = "CHAOTIC"
    RECOVERING = "RECOVERING"
    RUINED = "RUINED"


def local_chaos_pressure(disruption_intensity: float, congestion_memory: float, local_failed_qdot_ratio: float) -> float:
    return max(0.0, disruption_intensity + congestion_memory + local_failed_qdot_ratio)


def global_chaos_pressure(local_pressures: list[float], late_ratio: float) -> float:
    if not local_pressures:
        return late_ratio
    return mean(local_pressures) + late_ratio


def local_order_pressure(active_qdots: int, moving_qdots: int, failed_qdots: int) -> float:
    return max(0.0, float(active_qdots + moving_qdots - failed_qdots))


def global_order_pressure(active_qdots: int, failed_qdots: int, max_capacity: int, initial_qdots: int) -> float:
    utilization = active_qdots / max(max_capacity, 1)
    failure_drag = failed_qdots / max(initial_qdots, 1)
    return utilization - failure_drag


def choose_d_state(
    system_ruined: bool,
    current_chaos: float,
    previous_chaos: float,
    stressed_threshold: float,
    disrupted_threshold: float,
    chaotic_threshold: float,
    recovering_decay_ratio: float,
) -> DState:
    if system_ruined:
        return DState.RUINED
    if previous_chaos > 0 and current_chaos < previous_chaos * (1.0 - recovering_decay_ratio):
        return DState.RECOVERING
    if current_chaos >= chaotic_threshold:
        return DState.CHAOTIC
    if current_chaos >= disrupted_threshold:
        return DState.DISRUPTED
    if current_chaos >= stressed_threshold:
        return DState.STRESSED
    return DState.STABLE
