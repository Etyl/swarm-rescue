"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
import numpy as np
from typing import Optional

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import normalize_angle

class DroneWaypoint(DroneAbstract):
    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)
        self.counterStraight = 0
        self.angleStopTurning = random.uniform(-math.pi, math.pi)
        self.distStopStraight = random.uniform(10, 50)
        self.isTurning = False

        self.onRoute = True # True if the drone is on the route to the waypoint
        self.path = [[-320,-180],[-200,150],[20, -200]]
        self.nextWaypoint = self.path.pop() # The next waypoint to go to

    def define_message_for_all(self):
        """
        Here, we don't need communication...
        """
        pass

    def process_lidar_sensor(self):
        """
        Returns True if the drone collided an obstacle
        """
        if self.lidar_values() is None:
            return False

        collided = False
        dist = min(self.lidar_values())

        if dist < 40:
            collided = True

        return collided

    def adapt_angle_direction(self, pos):
        if self.measured_compass_angle() != None:
            
            def angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

            pos = np.array(pos)
            waypoint = np.array(self.nextWaypoint)
            waypoint_vect = waypoint - pos
            ref_vect = np.array([1, 0])

            
            drone_angle = normalize_angle(self.measured_compass_angle())
            waypoint_angle = normalize_angle(angle(ref_vect, waypoint_vect))

            angle = normalize_angle(waypoint_angle - drone_angle)

            if angle > 0.5:
                return 1.0
            if angle < -0.5:
                return -1.0
            if angle > 0:
                angle
            return angle
        return 0
    
    def check_waypoint(self, pos):
        if np.linalg.norm(pos - self.nextWaypoint) < 10:
            return True
        return False

    def control(self):

        gps_position = self.measured_gps_position()

        command = {"forward": 0.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}

        if self.onRoute:
            command["forward"] = 1.0
            command["rotation"] = self.adapt_angle_direction(gps_position)
            command["lateral"] = math.sin(command["rotation"])
            if self.check_waypoint(gps_position):
                if len(self.path) > 0:
                    self.nextWaypoint = self.path.pop()

                else:
                    self.nextWaypoint = None
                    self.onRoute = False
            
        return command
