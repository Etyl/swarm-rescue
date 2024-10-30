from __future__ import annotations

from typing import List, TYPE_CHECKING, Optional, Dict
import math
import numpy as np

from solutions.utils.types import Vector2D
from solutions.utils.utils import normalize_angle
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR

if TYPE_CHECKING:
    from solutions.frontier_drone import FrontierDrone

class Localizer:
    def __init__(self, drone: FrontierDrone):
        self.drone = drone
        self.theoretical_velocity: Vector2D = Vector2D()

    @property
    def drone_velocity(self) -> float:
        return self.drone.odometer_values()[0]

    @property
    def drone_velocity_angle(self) -> float:
        return normalize_angle(self.drone.odometer_values()[1])

    @property
    def drone_position(self) -> Vector2D:
        return self.drone.drone_position

    @property
    def drone_angle(self) -> float:
        return self.drone.drone_angle


    def get_angle_to_target(self) -> float:
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.drone.target is None:
            return 0

        drone_angle = normalize_angle(self.drone_angle)
        waypoint_angle = Vector2D(1,0).get_angle(self.drone.target - self.drone_position)

        return normalize_angle(waypoint_angle - drone_angle)


    def get_power(self) -> float:
        curr_velocity = self.drone.odometer_values()[0]
        dist_to_target = self.drone.drone_position.distance(self.drone.target)
        target_velocity = 3 + (1-math.exp(-dist_to_target/50))*7
        return math.tanh(target_velocity-curr_velocity)

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

        angle_from_target = self.get_angle_to_target() + self.drone.drone_angle_offset
        angle_from_target = normalize_angle(angle_from_target)
        power = self.get_power()

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


    def optimise_localization(self):
        """
        optimises the localization of the drone using SLAM
        """
        # starting_pos = self.drone_position
        # starting_angle = self.drone_angle
        starting_pos = self.drone.get_position()
        starting_angle = self.drone.get_angle()

        lidar_dists = self.drone.lidar().get_sensor_values()[::10].copy()
        lidar_angles = self.drone.lidar().ray_angles[::10].copy()
        measures = []
        for k in range(len(lidar_dists)):
            if lidar_dists[k] <= MAX_RANGE_LIDAR_SENSOR*0.7:
                measures.append([lidar_dists[k], lidar_angles[k]])
        def Q(x):
            [posX, posY, angle] = x
            value = 0
            for [lidar_dist,lidar_angle] in measures:
                point = Vector2D(0,0)
                point.setX(starting_pos.x + posX + lidar_dist*math.cos(lidar_angle+starting_angle+angle))
                point.setY(starting_pos.y + posY + lidar_dist*math.sin(lidar_angle+starting_angle+angle))
                point = self.drone.map.world_to_grid(point)
                if point.x < 0 or point.x >= self.drone.map.width or point.y < 0 or point.y >= self.drone.map.height:
                    continue
                #value -= self.map.occupancy_grid.get_grid()[int(point[0]),int(point[1])]
                value -= self.drone.map.get_confidence_wall_map(int(point.x),int(point.y))
            return value

        mindx, mindy, mindangle = 0,0,0

        for k in range(30):
            dx, dy, dangle = np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,0.1)
            if Q([dx,dy,dangle]) < Q([mindx,mindy,mindangle]):
                mindx, mindy, mindangle = dx, dy, dangle

        self.drone.drone_position = Vector2D(starting_pos.x + mindx, starting_pos.y + mindy)
        self.drone.drone_angle = starting_angle + mindangle


    # TODO: improve angle estimation
    def update_localization(self) -> None:
        """
        returns the position of the drone
        """

        rot = self.drone.command["rotation"]
        measured_angle = self.drone.measured_compass_angle()
        if measured_angle is not None:
            self.drone.drone_angle = measured_angle
            self.drone.time_in_no_gps = 0
        else:
            self.drone.drone_angle = self.drone.drone_angle + 0.2*rot
            #self.drone_angle += self.odometer_values()[2]

        measured_position = self.measured_gps_position()

        angle = self.drone.drone_angle
        command = Vector2D(self.drone.command["forward"], self.drone.command["lateral"])
        command.rotate(angle)

        theoretical_velocity = self.theoretical_velocity + ((command * 0.56) - (self.theoretical_velocity * 0.095))
        v = self.drone.odometer_values()[0]

        if measured_position is not None and abs(v) > 5:
            self.theoretical_velocity = Vector2D((v * math.cos(angle) + theoretical_velocity.x) / 2, (v * math.sin(angle) + theoretical_velocity.y) / 2)
            theoretical_position = self.drone.drone_position + self.theoretical_velocity
            self.drone.drone_position = (self.measured_gps_position() + theoretical_position) / 2
        elif measured_position is not None:
            self.theoretical_velocity = Vector2D(pointList=np.array([v * math.cos(angle), v * math.sin(angle)]) / 2)
            self.drone.drone_position =  self.measured_gps_position()
        else:
            self.drone.drone_position.setX(self.drone.drone_position.x + self.drone_velocity * np.cos(self.drone.drone_angle + self.drone_velocity_angle))
            self.drone.drone_position.setY(self.drone.drone_position.y + self.drone_velocity * np.sin(self.drone.drone_angle + self.drone_velocity_angle))
            # self.theoretical_velocity = theoretical_velocity
            # self.drone_position = self.drone_position + self.theoretical_velocity
            self.optimise_localization()

    def measured_gps_position(self) -> Optional[Vector2D]:
        measured_position = self.drone.measured_gps_position()
        if measured_position is None:
            return None
        return Vector2D(pointList=measured_position)