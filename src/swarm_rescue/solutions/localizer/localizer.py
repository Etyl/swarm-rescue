from __future__ import annotations

from typing import List, TYPE_CHECKING, Optional, Dict
import math

from solutions.utils.types import Vector2D
from solutions.utils.utils import normalize_angle

if TYPE_CHECKING: # type: ignore
    from solutions.frontier_drone import FrontierDrone # type: ignore

class Localizer:
    def __init__(self, drone: FrontierDrone):
        self.drone = drone

    def get_control_from_semantic(self):

        angular_vel_controller_max = 1.0

        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        if self.drone.found_wounded:
            best_angle = Vector2D(1, 0).get_angle(self.drone.wounded_target - self.drone.get_position())
            best_angle = normalize_angle(best_angle - normalize_angle(self.drone.get_angle()))

            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.4

        if self.drone.found_center and self.drone.center_angle is not None:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * self.drone.center_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if self.drone.near_center and self.drone.rescue_center_dist < 40:
            command["forward"] = 0
            command["rotation"] = 0.5
            command["lateral"] = 0.2

        return command

    # TODO updates params
    def get_control_from_path(self):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}

        self.drone.update_waypoint_index()
        if self.drone.next_waypoint is not None and self.drone.check_waypoint():
            if self.drone.check_waypoint():
                if self.drone.waypoint_index < len(self.drone.path)-1:
                    self.drone.waypoint_index += 1
                else:
                    self.drone.reset_path()

        if self.drone.target is None or self.drone.drone_position is None:
            return command

        angle_from_target = self.drone.adapt_angle_direction(self.drone.drone_position) + self.drone.drone_angle_offset
        angle_from_target = normalize_angle(angle_from_target)
        power = self.drone.get_power()

        if angle_from_target > 0.8:
            command["rotation"] =  1.0

        elif angle_from_target < -0.8:
            command["rotation"] =  -1.0
        else:
            command["rotation"] = angle_from_target

        angle_from_target = normalize_angle(angle_from_target - self.drone.drone_angle_offset)
        command["forward"] = power*math.cos(angle_from_target)
        command["lateral"] = power*math.sin(angle_from_target)

        return command