"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
import numpy as np
from typing import Optional
import arcade
from collections import deque 
from statemachine import exceptions
import scipy.optimize

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean, normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.mapper.mapper import Map
from solutions.localizer.localizer import Localizer
from solutions.roamer.roamer import RoamerController
from drone_controller import DroneController

class DroneWaypoint(DroneAbstract):

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

        self.onRoute = False # True if the drone is on the route to the waypoint
        self.path = []
        self.lastWaypoint = None # The last waypoint reached
        self.nextWaypoint = np.array([0,0]) # The next waypoint to go to
        self.drone_position = np.array([0,0]) # The position of the drone
        self.drone_angle = 0 # The angle of the drone
        self.found_wounded = False # True if the drone has found a wounded person
        self.found_center = False # True if the drone has found the rescue center
        self.command_semantic = None # The command to follow the wounded person or the rescue center
        self.last_angles = deque() # queue of the last angles

        self.wounded_found = []
        self.wounded_distance = 80 # The distance between wounded person to be considered as the same

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = True
        self.debug_positions = True
        self.debug_map = True
        
        # self.controller.force_transition()
        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        self.map = Map(area_world=self.size_area, resolution=8, lidar=self.lidar(), debug_mode=self.debug_map)
        self.rescue_center_position = None
        
        self.roamer_controller = RoamerController(self, self.map, debug_mode=False)

        self.localizer = Localizer()
        self.theorical_velocity = np.zeros(2)

        self.controller = DroneController(self, debug_mode=True)
        self.controller.force_transition()
        self.gps_disabled = True

        self.iteration = 0

        

    def adapt_angle_direction(self, pos: list):
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.drone_angle is not None and self.nextWaypoint is not None:
            
            def angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

            pos = np.array(pos)
            waypoint = np.array(self.nextWaypoint)
            waypoint_vect = waypoint - pos
            ref_vect = np.array([1, 0])

            
            drone_angle = normalize_angle(self.drone_angle)
            waypoint_angle = normalize_angle(angle(ref_vect, waypoint_vect))

            return normalize_angle(waypoint_angle - drone_angle)

        return 0
    
    
    def check_waypoint(self, pos):
        """
        checks if the drone has reached the waypoint
        """

        dist = np.linalg.norm(pos - self.nextWaypoint)
        if len(self.path) == 0: return dist < 20

        v1 = self.nextWaypoint - pos
        v2 = np.array(self.path[-1]) - np.array(self.nextWaypoint)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: 
            turning_angle = 0
        else:
            turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        # TODO: tune values
        return dist < 20 + (1+turning_angle)*20
        

    # TODO: implement communication
    def define_message_for_all(self):
        pass

    def get_position(self):
        """
        returns the position of the drone
        """
        return self.drone_position
    
    def get_angle(self):
        """
        returns the angle of the drone
        """
        return self.drone_angle


    def optimise_localization(self):
        """
        optimises the localization of the drone using SLAM
        """
        starting_pos = self.get_position()
        starting_angle = self.get_angle()

        lidar_dists = self.lidar().get_sensor_values()[::4].copy()
        lidar_angles = self.lidar().ray_angles[::4].copy()
        measures = []
        for k in range(len(lidar_dists)):
            if lidar_dists[k] <= MAX_RANGE_LIDAR_SENSOR*0.7:
                measures.append([lidar_dists[k], lidar_angles[k]])

        def Q(x):
            [posX, posY, angle] = x
            value = 0
            for [lidar_dist,lidar_angle] in measures:
                point = np.zeros(2)
                point[0] = starting_pos[0] + posX + lidar_dist*math.cos(lidar_angle+starting_angle+angle)
                point[1] = starting_pos[1] + posY + lidar_dist*math.sin(lidar_angle+starting_angle+angle)
                point = self.map.world_to_grid(point)
                if point[0] < 0 or point[0] >= self.map.x_max_grid or point[1] < 0 or point[1] >= self.map.y_max_grid:
                    continue
                value -= self.map.confidence_map[point[0],point[1]]
            return value
        
        """
        res = scipy.optimize.minimize(Q,
                                      np.array([starting_pos[0], starting_pos[1], starting_angle]),
                                      bounds=[(-4,4),(-4,4),(-0.2,0.2)],
                                      tol=1e-7)
        print(res.x, Q(res.x),Q([0,0,0]))
        dx,dy,dangle = res.x
        """
        mindx, mindy, mindangle = -10,-10,-0.2
        # for dx in np.linspace(-3,3,10):
        #     for dy in np.linspace(-3,3,10):
        #         for dangle in np.linspace(-0.1,0.1,10):
        #             if Q([dx,dy,dangle]) < Q([mindx,mindy,mindangle]):
        #                 mindx, mindy, mindangle = dx, dy, dangle

        for k in range(30):
            dx, dy, dangle = np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,0.1)
            if Q([dx,dy,dangle]) < Q([mindx,mindy,mindangle]):
                mindx, mindy, mindangle = dx, dy, dangle

        self.drone_position = np.array([starting_pos[0]+mindx, starting_pos[1]+mindy])
        self.drone_angle = starting_angle + mindangle


    # TODO: improve angle estimation
    def get_localization(self):
        """
        returns the position of the drone
        """

        rot = self.command["rotation"]
        measured_angle = self.measured_compass_angle()
        if measured_angle is not None:
            self.drone_angle = measured_angle
            self.gps_disabled = False
        else:
            self.drone_angle = self.drone_angle + 0.2*rot
            self.gps_disabled = True

        measured_position = self.measured_gps_position()

        angle = self.drone_angle
        rot_matrix = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
        command = np.array([self.command["forward"], self.command["lateral"]])
        command = command@rot_matrix

        theorical_velocity = self.theorical_velocity + (command*0.56 - self.theorical_velocity*0.095)
        v = self.odometer_values()[0]

        if measured_position is not None and abs(v) > 5:  
            self.theorical_velocity = (np.array([v*math.cos(angle), v*math.sin(angle)]) + theorical_velocity) / 2
            theoretical_position = self.drone_position + self.theorical_velocity 
            self.drone_position = (self.measured_gps_position() + theoretical_position)/2
        elif measured_position is not None:
            self.theorical_velocity = np.array([v*math.cos(angle), v*math.sin(angle)]) / 2
            self.drone_position = self.measured_gps_position()
        else:
            self.theorical_velocity = theorical_velocity
            self.drone_position = self.drone_position + self.theorical_velocity
            self.optimise_localization()


    def add_wounded(self, data_wounded):
        """
        compares the wounded persons detected with the ones already detected
        """

        def get_wounded_position():
            wounded_pos = self.drone_position.copy()
            angle = normalize_angle(self.drone_angle + data_wounded.angle)
            wounded_pos[0] += data_wounded.distance * math.cos(angle)
            wounded_pos[1] += data_wounded.distance * math.sin(angle)
            return wounded_pos

        wounded_pos = get_wounded_position()
    
        for k in range(len(self.wounded_found)):
            wounded = self.wounded_found[k]
            if np.linalg.norm(wounded_pos - wounded["position"]) < self.wounded_distance:
                wounded["count"] += 1
                n = wounded["count"]
                wounded["position"] = wounded["position"]*((n-1)/n) + wounded_pos/n
                wounded["last_seen"] = 0
                return
            
        self.wounded_found.append({"position": wounded_pos, "count": 1, "last_seen": 0})
    
    def clear_wounded_found(self):
        frame_limit = 10

        for k in range(len(self.wounded_found)-1,-1,-1):
            self.wounded_found[k]["last_seen"] += 1
            if np.linalg.norm(self.drone_position - self.wounded_found[k]["position"])<80 and self.wounded_found[k]["last_seen"] > frame_limit:
                self.wounded_found.pop(k)

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        
        best_angle = 0

        self.clear_wounded_found()

        found_wounded = False
        if (detection_semantic):
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    self.add_wounded(data)
            
            found_wounded = len(self.wounded_found) > 0
            if (len(self.wounded_found) > 0 and 
                (self.controller.current_state == self.controller.going_to_wounded 
                or self.controller.current_state == self.controller.approaching_wounded
                or self.controller.current_state == self.controller.roaming)):
            
                # Select the best one among wounded persons detected
                min_distance = np.linalg.norm(self.drone_position - self.wounded_found[0]["position"])
                best_position = self.wounded_found[0]["position"]
                for wounded in self.wounded_found[1:]:
                    distance = np.linalg.norm(self.drone_position - wounded["position"])
                    if distance < min_distance:
                        min_distance = distance
                        best_position = wounded["position"]
                best_angle = normalize_angle(angle(np.array([1,0]), best_position - self.drone_position))
                best_angle = normalize_angle(best_angle - normalize_angle(self.drone_angle))

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (detection_semantic and 
            (self.controller.current_state == self.controller.going_to_center 
            or self.controller.current_state == self.controller.approaching_center)):
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))


        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command


    # TODO: implement pathfinder with map
    def get_path(self, pos, destination):
        """
        returns the path to the destination
        """
        if destination == 0:
            return [[-320,-180],[-260,138],[-200,150], [20, -200]]
        
        path = self.map.shortest_path(self.drone_position, pos)
        return path


    def get_control_from_path(self, pos):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angle_from_waypoint = self.adapt_angle_direction(pos)
        angle_from_waypoint = normalize_angle(angle_from_waypoint)

        if angle_from_waypoint > 0.5:
            command["rotation"] =  1.0
        elif angle_from_waypoint < -0.5:
            command["rotation"] =  -1.0
        else:
            command["rotation"] = angle_from_waypoint

        command["forward"] = math.cos(angle_from_waypoint)
        command["lateral"] = math.sin(angle_from_waypoint)
        norm = max(abs(command["forward"]),abs(command["lateral"]))
        command["forward"] = command["forward"]/norm
        command["lateral"] = command["lateral"]/norm     
        
        if self.nextWaypoint is not None and self.check_waypoint(pos):
            if self.check_waypoint(pos):
                if len(self.path) > 0:
                    self.lastWaypoint = self.nextWaypoint.copy()
                    self.nextWaypoint = self.path.pop()
                else:
                    self.nextWaypoint = None
                    self.onRoute = False
        return command
    
    def compute_rescue_center_position(self):
        """
        computes the position of the rescue center
        """
        semantic_lidar_dist = [data.distance for data in self.semantic_values() if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER]
        min_dist = min(semantic_lidar_dist) if len(semantic_lidar_dist) > 0 else np.inf
        
        if min_dist > 10:
            self.rescue_center_position = self.drone_position.copy()

    def control(self):
        self.iteration += 1
        self.found_wounded, self.found_center, self.command_semantic = self.process_semantic_sensor()
        self.get_localization()
        
        if self.rescue_center_position is None:
            self.compute_rescue_center_position()
        

        if self.roaming:
            try:
                self.roamer_controller.cycle()
            except exceptions.TransitionNotAllowed:
                pass
        
        self.controller.cycle()
        self.update_mapping()
        self.keep_distance_from_walls()
            
        if self.roaming:
            if self.gps_disabled:
               self.roamer_controller.command["rotation"] /=2
               self.roamer_controller.command["forward"] /=2
               self.roamer_controller.command["lateral"] /=2
            self.command = self.roamer_controller.command.copy()
        else:
            if self.gps_disabled:
               self.controller.command["rotation"] /=2
               self.controller.command["forward"] /=2
               self.controller.command["lateral"] /=2
            self.command = self.controller.command.copy()
        return self.command
    
    def keep_distance_from_walls(self):
        """
        keeps the drone at a distance from the walls
        """
        
        lidar_dist = self.lidar().get_sensor_values()
        lidar_angles = self.lidar().ray_angles

        min_dist_index = np.argmin(lidar_dist)
        min_dist = lidar_dist[min_dist_index]
        min_angle = lidar_angles[min_dist_index]

        if min_dist < 30:
            return min_angle, min_dist
        return None
    
    def update_mapping(self):
        """
        updates the map
        """
        detection_semantic = self.semantic_values()
        self.estimated_pose = Pose(self.drone_position, self.drone_angle)
        max_vel_angle = 0.08
        if abs(self.measured_angular_velocity()) < max_vel_angle:
            self.map.update(self.estimated_pose, detection_semantic)


    def draw_top_layer(self):

        if True:
            lidar_dist = self.lidar().get_sensor_values()[::].copy()
            lidar_angles = self.lidar().ray_angles[::].copy()
            for k in range(len(lidar_dist)):
                pos = self.get_position() + np.array(self.size_area)/2
                pos[0] += lidar_dist[k]*math.cos(lidar_angles[k]+self.get_angle())
                pos[1] += lidar_dist[k]*math.sin(lidar_angles[k]+self.get_angle())
                arcade.draw_circle_filled(pos[0], pos[1],2, arcade.color.PURPLE)


        if self.debug_wounded:
            for wounded in self.wounded_found:
                pos = np.array(wounded["position"]) + np.array(self.size_area)/2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.GREEN_YELLOW)
                arcade.draw_circle_outline(pos[0], pos[1],self.wounded_distance, arcade.color.GREEN_YELLOW)

        if self.debug_positions:
            pos = np.array(self.drone_position) + np.array(self.size_area)/2
            arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.RED)

            direction = np.array([1,0])
            rot = np.array([[math.cos(self.drone_angle), math.sin(self.drone_angle)],[-math.sin(self.drone_angle), math.cos(self.drone_angle)]])
            direction = direction@rot
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*200, pos[1]+direction[1]*200, arcade.color.RED)

            direction = self.theorical_velocity
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*20, pos[1]+direction[1]*20, arcade.color.GREEN)


    def draw_bottom_layer(self):

        if self.debug_path: 
            drawn_path = self.path.copy()
            if self.nextWaypoint is not None: drawn_path.append(self.nextWaypoint)
            if self.lastWaypoint != None: drawn_path.append(self.lastWaypoint)
            for k in range(len(drawn_path)-1):
                pt1 = np.array(drawn_path[k]) + np.array(self.size_area)/2
                pt2 = np.array(drawn_path[k+1]) + np.array(self.size_area)/2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 255))
