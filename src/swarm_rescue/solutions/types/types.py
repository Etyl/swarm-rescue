import numpy as np
from solutions.mapper.mapper import Map

class DroneData:
    def __init__(self) -> None:
        self.id: int = None
        self.position: np.ndarray = None
        self.angle: float = None
        self.vel_angle: float = None
        self.wounded_found: list = None
        self.wounded_target: np.ndarray = None
        self.map: Map = None
        self.semantic_values: np.ndarray = None
        self.kill_zone_mode: bool = None
