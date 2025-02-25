import math
import typing
from typing import Optional, List, Union, Dict, Tuple
import numpy as np
from dataclasses import dataclass

from scipy.constants import point


class Vector2D:
    def __init__(self, x : float = 0, y: float = 0, pointList : Union[Tuple,List,np.ndarray,None] = None):
        self.array: np.ndarray[float]
        if pointList is not None:
            self.array = np.array([pointList[0], pointList[1]])
        else:
            self.array = np.array([x, y])

    def __str__(self) -> str:
        return f"{self.array[0]};{self.array[1]}"

    def __add__(self, other):
        return Vector2D(pointList=(self.array + other.array))

    def __sub__(self, other):
        return Vector2D(pointList=(self.array - other.array))

    def __mul__(self, other):
        return Vector2D(pointList=(self.array * other))

    def __rmul__(self, other):
        return Vector2D(pointList=(other * self.array))

    def __truediv__(self, other):
        return Vector2D(pointList=(self.array / other))

    def __neg__(self):
        return Vector2D(pointList=(self.array * -1))

    def __matmul__(self, other: 'Vector2D') -> float:
        return self.array @ other.array

    def __rmatmul__(self, other: 'Vector2D') -> float:
        return other.array @ other.array

    def distance(self, position) -> float:
        return np.linalg.norm(self.array - position.array)

    def norm(self) -> float:
        return np.linalg.norm(self.array)

    def normalize(self) -> 'Vector2D':
        if self.norm() > 0:
            return Vector2D(pointList=self.array/self.norm())
        return Vector2D(0,0)

    def copy(self):
        return Vector2D(pointList=self.array)

    def rotate(self, angle:float) -> 'Vector2D':
        rot_matrix = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        return Vector2D(pointList=self.array @ rot_matrix)

    def get_angle(self, other: 'Vector2D') -> float:
        if self.norm()==0 or other.norm()==0:
            return 0
        v1 = self.normalize()
        v2 = other.normalize()
        return (math.atan2(v2.y,v2.x) - math.atan2(v1.y,v1.x)) % (2 * np.pi)

    def project(self, v1: 'Vector2D', v2: 'Vector2D') -> 'Vector2D':
        """
        Returns the projected vector on the line defined by v1 and v2
        """
        if v1 == v2: return self.copy()
        x1: Vector2D = self-v1
        x2: Vector2D = (v2-v1).normalize()
        return v1 + (x1@x2)*x2

    @property
    def x(self):
        return self.array[0]

    @property
    def y(self):
        return self.array[1]

    def setX(self, x):
        self.array[0] = x

    def setY(self, y):
        self.array[1] = y


@dataclass
class WoundedData:
    position: Vector2D
    count : int
    last_seen : int
    drone_taker: Optional[int]


@dataclass
class DroneData:
    id: int
    position: Vector2D
    angle: float
    vel_angle: float
    wounded_target: Optional[Vector2D]
    map : typing.Any
    semantic_values: np.ndarray
    next_waypoint : Vector2D
    confidence_position: float
    visible_drones: List[Tuple[int, Vector2D]]
    rescue_center_position: Optional[Vector2D]
    no_comm_zone_mode: bool
    target : Optional[Vector2D]
    drone_positions: Dict[int,Tuple[int,Vector2D]] # Dict of drone id -> timestep,drone_position