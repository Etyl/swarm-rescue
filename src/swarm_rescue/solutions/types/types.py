import math
import typing
from typing import Optional, List, Union
import numpy as np

import solutions
from dataclasses import dataclass


class Vector2D:
    def __init__(self, x : float = 0, y: float = 0, pointList : Optional[Union[List,np.ndarray]] = None):
        self.array: np.ndarray[float]
        if pointList is not None:
            self.array = np.array([pointList[0], pointList[1]])
        else:
            self.array = np.array([x, y])

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

    def dot(self, other: 'Vector2D') -> float:
        return self.array @ other.array

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

    def rotate(self, angle:float) -> None:
        rot_matrix = np.array([[math.cos(angle), math.sin(angle)], [-math.sin(angle), math.cos(angle)]])
        self.array = self.array @ rot_matrix

    def get_angle(self, other: 'Vector2D') -> float:
        if self.norm()==0 or other.norm()==0:
            return 0
        v1 = self.normalize()
        v2 = other.normalize()
        return math.atan2(v2.y,v2.x) - math.atan2(v1.y,v1.x)

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
    wounded_target: Vector2D
    map : typing.Any
    semantic_values: np.ndarray
    kill_zone_mode: bool
    nextWaypoint : Vector2D