from __future__ import annotations

from collections import deque
from typing import TYPE_CHECKING, Optional, Deque, Dict, List, Tuple
import math
import numpy as np
import os

from solutions.utils.types import Vector2D
from solutions.utils.utils import normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR
from spg_overlay.utils.utils import circular_mean

if TYPE_CHECKING:
    from solutions.frontier_drone import FrontierDrone

INF: float = 100000

def get_acceleration(velocity: Vector2D, command, drone_angle: float) -> Vector2D:
    B = Vector2D(command["forward"], command["lateral"]).rotate(drone_angle)
    if B.norm()==0:
        return Vector2D(0,0)
    if B.norm() > 1:
        B = B.normalize()
    B = Localizer.velocity_const * B
    return Vector2D((B.x - velocity.x) / Localizer.tau, (B.y - velocity.y) / Localizer.tau)


def get_angular_velocity(command) -> float:
    return Localizer.angle_const * command["rotation"]

def angle_distance(a1, a2)-> float:
    a1 = normalize_angle(a1)
    a2 = normalize_angle(a2)
    r = abs(a1 - a2)
    r = min(r, abs(a1-a2+2*np.pi))
    r = min(r, abs(a1-a2-2*np.pi))
    return r

class Localizer:
    velocity_const = 11.72052117358362
    angle_const = 0.171
    tau = 20
    queue_size = 3

    def __init__(self, drone: FrontierDrone):
        self.drone = drone
        self.last_impact: int = 10000
        self.confidence_position = 0
        self.time_approaching_center = 0

        self._drone_velocity: Vector2D = Vector2D()
        self._drone_velocity_angle: float = 0
        self._drone_position: Vector2D = Vector2D()
        self._drone_angle: float = 0

        self._measured_position: Optional[Vector2D] = None
        self._measured_velocity: Optional[Vector2D] = None
        self._measured_angle: Optional[float] = None
        self._measured_velocity_angle: Optional[float] = None

        self._previous_velocity: Deque[Vector2D] = deque([])
        self._previous_velocity_angle: Deque[float] = deque()
        self._previous_position: Deque[Vector2D] = deque([])
        self._previous_angle: Deque[float] = deque([])

        self._theoretical_position: Vector2D = Vector2D(INF,INF)
        self._theoretical_velocity: Vector2D = Vector2D(INF,INF)
        self._theoretical_angle: float = 0
        self._theoretical_angle_velocity: float = 0

    @property
    def drone_speed(self) -> float:
        return self.drone.odometer_values()[0]

    @property
    def drone_speed_alpha(self) -> float:
        return self.drone.odometer_values()[1]

    @property
    def drone_velocity(self) -> Vector2D:
        return self._drone_velocity

    @property
    def drone_velocity_angle(self) -> float:
        return self._drone_velocity_angle

    @property
    def drone_position(self) -> Vector2D:
        return self._drone_position

    @property
    def drone_angle(self) -> float:
        return self._drone_angle

    @property
    def theoretical_position(self) -> Vector2D:
        return self._theoretical_position

    @property
    def theoretical_velocity(self) -> Vector2D:
        return self._theoretical_velocity

    @property
    def measured_position(self) -> Vector2D:
        return self._measured_position

    def update_previous(self) -> None:
        if self._measured_velocity is not None:
            self._previous_velocity.append(self._measured_velocity)
        if self._measured_velocity_angle is not None:
            self._previous_velocity_angle.append(self._measured_velocity_angle)
        if self._measured_position is not None:
            self._previous_position.append(self._measured_position)
        if self._measured_velocity_angle is not None:
            self._previous_angle.append(self._measured_angle)

        if len(self._previous_position)>Localizer.queue_size or (self._measured_position is None and len(self._previous_position)>0):
            self._previous_position.popleft()
        if len(self._previous_velocity)>Localizer.queue_size or (self._measured_velocity is None and len(self._previous_velocity)>0):
            self._previous_velocity.popleft()
        if len(self._previous_velocity_angle)>Localizer.queue_size or (self._measured_velocity_angle is None and len(self._previous_velocity_angle)>0):
            self._previous_velocity_angle.popleft()
        if len(self._previous_angle)>Localizer.queue_size or (self._measured_angle is None and len(self._previous_angle)>0):
            self._previous_angle.popleft()

    def update_measured_values(self) -> None:
        if self.drone.measured_gps_position() is None:
            self._measured_position = None
        else:
            self._measured_position = Vector2D(pointList=self.drone.measured_gps_position())

        if self.drone.measured_velocity() is None:
            self._measured_velocity = None
        else:
            self._measured_velocity = Vector2D(pointList=self.drone.measured_velocity())

        self._measured_angle = self.drone.measured_compass_angle()
        self._measured_velocity_angle = self.drone.measured_angular_velocity()

    def update_theoretical_values(self) -> None:
        """
        Called before estimating drone values (=> previous drone values)
        """
        previous_command = self.drone.prev_command
        if self._measured_velocity is not None and (self.last_impact<=Localizer.queue_size or self.theoretical_velocity.distance(self._measured_velocity)>10):
            self._theoretical_velocity = self._measured_velocity
        else:
            a = get_acceleration(self._drone_velocity, previous_command, self._drone_angle)
            self._theoretical_velocity = self._drone_velocity + a

        if self._measured_position is not None and (self.last_impact<=Localizer.queue_size or self._theoretical_position.distance(self._measured_position)>20):
            self._theoretical_position = self._measured_position
        else:
            self._theoretical_position = self._drone_position + self._theoretical_velocity

        self._theoretical_angle_velocity = get_angular_velocity(previous_command)
        self._theoretical_angle = self._drone_angle + self._theoretical_angle_velocity


    def localize(self) -> None:
        if self.drone.has_collided:
            self.last_impact = 0
        else:
            self.last_impact += 1

        self.update_measured_values()
        self.update_theoretical_values()
        self.update_previous()

        # Estimate drone angle
        if self._measured_angle is not None:
            self._drone_angle = self._measured_angle
        else:
            self._drone_angle = self._theoretical_angle

        # Estimate drone velocity
        if self._measured_velocity is not None and self.last_impact<=Localizer.queue_size:
            self._drone_velocity = self._measured_velocity
        elif self._measured_velocity is not None:
            last_v: Vector2D = Vector2D()
            for v in self._previous_velocity:
                last_v += v
            last_v = last_v / len(self._previous_velocity)
            self._drone_velocity = last_v + get_acceleration(last_v, self.drone.prev_command, self._drone_angle)
        else:
            self._drone_velocity = self._theoretical_velocity

        # Estimate drone position
        if self._measured_position is not None:
            last_pos: Vector2D = Vector2D()
            self.confidence_position = 1
            for p in self._previous_position:
                last_pos += p
            last_pos = last_pos / len(self._previous_position)
            self._drone_position = last_pos + self._drone_velocity
        else: # TODO improve
            self._drone_position += self.drone_speed * Vector2D(math.cos(self.drone_angle+self.drone_speed_alpha), math.sin(self.drone_angle+self.drone_speed_alpha))
            self.optimise_from_scan_match()
            # self.optimize_from_communicator()

        # path = os.path.dirname(os.path.abspath(__file__))
        # with open(path+"./data/measured_pos.txt", 'a') as f:
        #     f.write(f"{self.drone.true_position()[0]} {self.drone.true_position()[1]} "+
        #             f"{self._drone_position.x} {self._drone_position.y} "+
        #             f"{self._measured_position.x} {self._measured_position.y} "+
        #             f"{self.drone.true_velocity()[0]} {self.drone.true_velocity()[1]} "+
        #             f"{self.drone_velocity.x} {self.drone_velocity.y} "+
        #             f"{self._measured_velocity.x} {self._measured_velocity.y} "+
        #             f"{self.drone.true_angle()} {self._drone_angle} {self._measured_angle}"
        #             '\n')


    # NOT USED
    def estimate_confidence(self, drone_confidences:List[float]) -> float:
        p = self.confidence_position
        n = 1
        for x in drone_confidences:
            if x>self.confidence_position:
                p += x
                n += 1
        p /= n

        return p

    # NOT USED
    def optimize_from_communicator(self) -> None:
        drone_positions: List[Tuple[float,Vector2D]] = []
        data_list = self.drone.communicator.received_messages
        for (drone_communicator, drone_data) in data_list:
            found: List[Vector2D] = []
            for (x,y) in drone_data.visible_drones:
                if x==self.drone.identifier:
                    found.append(y)
            if len(found)==0:
                continue
            pos: Vector2D = Vector2D()
            for p in found: pos += p
            pos /= len(found)
            drone_positions.append((drone_data.confidence_position,pos))


        estimated_pos = self.confidence_position * self.drone_position
        drones_confidences = [self.confidence_position]
        for (c,pos) in drone_positions:
            if c<self.confidence_position:
                continue
            estimated_pos += c * pos
            drones_confidences.append(c)

        estimated_pos /= sum(drones_confidences)

        self.confidence_position = self.estimate_confidence(drones_confidences)
        self._drone_position = estimated_pos


    def optimise_from_scan_match(self):
        """
        optimises the localization of the drone using SLAM
        """
        # starting_pos = self.drone_position
        # starting_angle = self.drone_angle
        starting_pos = self.drone_position
        starting_angle = self.drone_angle

        lidar_dists = self.drone.lidar().get_sensor_values()[::10].copy()
        lidar_angles = self.drone.lidar().ray_angles[::10].copy()
        measures = []
        for k in range(len(lidar_dists)):
            if lidar_dists[k] <= MAX_RANGE_LIDAR_SENSOR*0.7:
                measures.append([lidar_dists[k], lidar_angles[k]])
        def confidence(x):
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
                value += self.drone.map.get_confidence_wall_map(int(point.x),int(point.y))
            return value / len(measures)

        dx_best, dy_best, dangle_best = 0,0,0
        max_confidence = confidence([dx_best,dy_best,dangle_best])
        for k in range(30):
            dx, dy, dangle = np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,0.1)
            new_confidence = confidence([dx,dy,dangle])
            if new_confidence > max_confidence:
                dx_best, dy_best, dangle_best = dx, dy, dangle
                max_confidence = new_confidence

        self._drone_position = starting_pos + Vector2D(dx_best,dy_best)
        self._drone_angle = starting_angle + dangle_best
        self.confidence_position = max_confidence


    # ================== PATH PLANNING ==================

    def get_angle_to_target(self, target:Vector2D) -> float:
        """
        gives the angle to turn to in order to go to the next waypoint
        """
        drone_angle = normalize_angle(self.drone_angle)
        waypoint_angle = Vector2D(1,0).get_angle(target - self.drone_position)

        return normalize_angle(waypoint_angle - drone_angle)


    def get_power(self, target: Vector2D) -> float:
        curr_velocity = self.drone.odometer_values()[0]
        dist_to_target = self.drone.drone_position.distance(target)
        target_velocity = 3 + (1-math.exp(-dist_to_target/50))*7
        return math.tanh(target_velocity-curr_velocity)

    def get_control_from_target(self, target: Vector2D) -> Dict[str, float]:
        command = {
            "forward": 1.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }

        angle_from_target = self.get_angle_to_target(target) + self.drone.drone_angle_offset
        angle_from_target = normalize_angle(angle_from_target)
        power = self.get_power(target)

        if angle_from_target > 0.8:
            command["rotation"] = 1.0

        elif angle_from_target < -0.8:
            command["rotation"] = -1.0
        else:
            command["rotation"] = angle_from_target

        angle_from_target = normalize_angle(angle_from_target - self.drone.drone_angle_offset)
        command["forward"] = power * math.cos(angle_from_target)
        command["lateral"] = power * math.sin(angle_from_target)

        return command


    def get_control_from_semantic(self):

        command = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }

        if self.drone.wounded_visible and self.drone.wounded_target is not None:
            command = self.get_control_from_target(self.drone.wounded_target)
            command["forward"] /= 2
            command["lateral"] /= 2

        if self.drone.found_center and self.drone.center_angle is not None:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * self.drone.center_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a
            command["forward"] = 1

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if self.drone.near_center and self.drone.rescue_center_dist < 40:
            command["forward"] = 0
            command["rotation"] = 0
            if self.time_approaching_center%160 < 20:
                command["forward"] = 0.2
            elif self.time_approaching_center%160 < 80:
                command["rotation"] = 0.5
            elif self.time_approaching_center%160 < 100:
                command["forward"] = 0.2
            else:
                command["rotation"] = -0.5
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
                    self.drone.update_waypoint_index()
                else:
                    self.drone.reset_path()

        if self.drone.target is None or self.drone.drone_position is None:
            self.drone.reset_path()
            return command

        command = self.get_control_from_target(self.drone.target)

        return command