from typing import Optional

import numpy as np
from solutions.mapper.mapper import Map
from dataclasses import dataclass

@dataclass
class DroneData:
    id: int
    position: np.ndarray
    angle: float
    vel_angle: float
    wounded_target: np.ndarray
    map: Map
    semantic_values: np.ndarray
    kill_zone_mode: bool
    nextWaypoint : np.ndarray


@dataclass
class WoundedData:
    position: np.ndarray
    count : int
    last_seen : int
    drone_taker: Optional[int]
