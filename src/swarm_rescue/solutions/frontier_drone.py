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

import time

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean, normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.mapper.mapper import Map
from solutions.localizer.localizer import Localizer
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR
from solutions.types.types import DroneData
import solutions


class FrontierDrone(DroneAbstract):

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
        self.drone_angle_offset = 0 # The angle offset of the drone that can be changed by the states
        self.found_wounded = False # True if the drone has found a wounded person
        self.found_center = False # True if the drone has found the rescue center
        self.command_semantic = None # The command to follow the wounded person or the rescue center
        self.last_angles = deque() # queue of the last angles
        self.repulsion = np.zeros(2) # The repulsion vector

        self.wounded_found = [] # the list of wounded persons found
        self.wounded_distance = 80 # The distance between wounded person to be considered as the same
        self.wounded_target = None # The wounded position to go to

        self.drone_list = [] # The list of drones

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = False
        self.debug_positions = False
        self.debug_map = False
        self.debug_roamer = False
        self.debug_controller = False 
        self.debug_lidar = False
        self.debug_repulsion = True
        self.debug_kill_zones = True
        
        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        #self.map = Map(area_world=self.size_area, resolution=8, lidar=self.lidar(), debug_mode=self.debug_map)
        self.map = Map(drone=self, area_world=self.size_area, drone_lidar=self.lidar(), resolution=10, identifier=self.identifier, debug_mode=True)
        self.rescue_center_position = None
        
        self.roamer_controller = solutions.roamer.RoamerController(self, self.map, debug_mode=self.debug_roamer)

        self.localizer = Localizer()
        self.theorical_velocity = np.zeros(2)

        self.controller = solutions.drone_controller.DroneController(self, debug_mode=self.debug_controller)
        self.controller.force_transition()
        self.gps_disabled = True

        self.last_other_drones_position = {}
        self.kill_zones = []
        self.semantic_drone_pos = []
        self.potential_kill_zones = []

        self.iteration = 0


    def adapt_angle_direction(self, pos: list):
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.get_angle() is not None and self.nextWaypoint is not None:
            
            def angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

            pos = np.array(pos)
            waypoint = np.array(self.nextWaypoint)
            waypoint_vect = waypoint - pos
            ref_vect = np.array([1, 0])
 
            drone_angle = normalize_angle(self.get_angle())
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
        

    def define_message_for_all(self):
        data = DroneData()
        data.id = self.identifier
        data.position = self.get_position()
        data.angle = self.get_angle()
        data.vel_angle = self.compute_vel_angle()
        data.wounded_found = self.wounded_found
        data.wounded_target = self.wounded_target
        data.map = self.map
        data.semantic_values = self.semantic_values()
        return data

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

        lidar_dists = self.lidar().get_sensor_values()[::10].copy()
        lidar_angles = self.lidar().ray_angles[::10].copy()
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
                if point[0] < 0 or point[0] >= self.map.width or point[1] < 0 or point[1] >= self.map.height:
                    continue
                value -= self.map.occupancy_grid.get_grid()[int(point[0]),int(point[1])]
            return value
        
        mindx, mindy, mindangle = 0,0,0

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
            angle = normalize_angle(self.get_angle() + data_wounded.angle)
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

    def update_drones(self, drone_data : DroneData):
        """
        updates the data of the drones
        """

        if drone_data.wounded_target is not None:
            for k in range(len(self.wounded_found)):
                if np.linalg.norm(self.wounded_found[k]["position"] - drone_data.wounded_target) < self.wounded_distance:
                    self.wounded_found[k]["taken"] = True
                    break
        for k in range(len(self.drone_list)):
            if self.drone_list[k].id == drone_data.id:
                self.drone_list[k] = drone_data
                return
        self.drone_list.append(drone_data)


    def process_communicator(self):
        """
        process the information from the communicator
        """
        data_list  = self.communicator.received_messages
        self.drone_list = []
        for (drone_communicator,drone_data) in data_list:
            self.update_drones(drone_data)


    def check_wounded_available(self):

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        self.found_wounded = False
        if (len(self.wounded_found) > 0 and 
            (self.controller.current_state == self.controller.going_to_wounded 
            or self.controller.current_state == self.controller.approaching_wounded
            or self.controller.current_state == self.controller.roaming)):
        
            # Select the best one among wounded persons detected
            min_distance = 0
            best_position = None
            for wounded in self.wounded_found:
                distance = np.linalg.norm(self.get_position() - wounded["position"])
                if "taken" not in wounded and (best_position is None or distance < min_distance):
                    min_distance = distance
                    best_position = wounded["position"]
            self.wounded_target = best_position
            if best_position is not None:
                self.found_wounded = True
        
        if self.found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0

            command = {"forward": 1.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

            best_angle = normalize_angle(angle(np.array([1,0]), best_position - self.drone_position))
            best_angle = normalize_angle(best_angle - normalize_angle(self.get_angle()))

            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.4
            
            self.command_semantic = command



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

        if (detection_semantic):
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    self.add_wounded(data)

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


        if found_rescue_center:
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

        return found_rescue_center, command


    def get_path(self, pos):
        """
        returns the path to the destination
        """
        return self.map.shortest_path(self.drone_position, pos)
        

    def get_control_from_path(self, pos):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angle_from_waypoint = self.adapt_angle_direction(pos) + self.drone_angle_offset
        angle_from_waypoint = normalize_angle(angle_from_waypoint)

        if angle_from_waypoint > 0.8:
            command["rotation"] =  1.0
        elif angle_from_waypoint < -0.8:
            command["rotation"] =  -1.0
        else:
            command["rotation"] = angle_from_waypoint

        angle_from_waypoint = normalize_angle(angle_from_waypoint - self.drone_angle_offset)
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


    def drone_repulsion(self):
        """
        repulses the drone from other drones
        Returns:
            repulsion float[2]: the repulsion vector
        """
        def compute_angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        def repulse_func(dist):
            return 10/max(1,(dist-60))
        
        repulsion = np.zeros(2)
        drone_pos = self.get_position()
        for drone in self.drone_list:
            dist = np.linalg.norm(drone_pos - drone.position)
            global_angle = normalize_angle(compute_angle(np.array([1,0]), drone.position - drone_pos))
            angle = normalize_angle(global_angle - self.get_angle())
            repulsion -= repulse_func(dist) * np.array([math.cos(angle), math.sin(angle)])

        if not self.roaming:
            repulsion /= 10

        self.repulsion = repulsion

    def compute_vel_angle(self):
        """
        compute the velocity angle
        """
        # use previous position to calculate velocity
        vel = self.measured_velocity()
        if vel is None:
            return 0
        vel_angle = math.atan2(vel[1], vel[0])
        return vel_angle
    
    def check_other_drones_killed(self):
        """
        checks if the other drones are killed
        """
        # update other drones distance
        drone_list_alt = []
        for drone in self.drone_list:
            #if np.linalg.norm(drone.position - self.drone_position) < RANGE_COMMUNICATION:
            drone_list_alt.append(drone)
        for drone in drone_list_alt:
            if drone.id == self.identifier: continue
            self.last_other_drones_position[drone.id] = [drone.position, drone.vel_angle]
        
        # check if other drones are killed by checking it's not in drone_list anymore and have last seen distance < MAX_RANGE_LIDAR_SENSOR / 2
        killed_ids = []
        for id in self.last_other_drones_position:
            if id not in [drone.id for drone in drone_list_alt]:
                if np.linalg.norm(self.last_other_drones_position[id][0] - self.drone_position) < RANGE_COMMUNICATION * 0.85:
                    #print(f"Drone {id} killed")
                    self.path = []
                    self.nextWaypoint = None
                    kill_zone_x = self.last_other_drones_position[id][0][0] + 50*math.cos(self.last_other_drones_position[id][1])
                    kill_zone_y = self.last_other_drones_position[id][0][1] + 50*math.sin(self.last_other_drones_position[id][1])
       
                    # is_kill_zone = True
                    # for drone in drone_list_alt:
                    #     for data in drone.semantic_values:
                    #         if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    #             drone_x = drone.position[0] + data.distance * math.cos(data.angle + drone.angle)
                    #             drone_y = drone.position[1] + data.distance * math.sin(data.angle + drone.angle)
                    #             if np.linalg.norm(np.array([drone_x, drone_y]) - np.array([self.last_other_drones_position[id][0][0],  self.last_other_drones_position[id][0][1]])) < 50:
                    #                 is_kill_zone = False
                    #--------------------
                    # for data in self.semantic_values():
                    #     if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    #         drone_x = self.drone_position[0] + data.distance * math.cos(data.angle + self.get_angle())
                    #         drone_y = self.drone_position[1] + data.distance * math.sin(data.angle + self.get_angle())
                    #         # count the number of positions near the drone in the last 50 frames
                    #         count = 0
                    #         for pos in self.semantic_drone_pos:
                    #             if np.linalg.norm(pos - np.array([self.last_other_drones_position[id][0][0], self.last_other_drones_position[id][0][1]])) < 5:
                    #                 count += 1
                    #         if count > 5:
                    #             self.map.add_kill_zone(id, [kill_zone_x, kill_zone_y])
                        #self.map.add_kill_zone(id, [kill_zone_x, kill_zone_y])
                    self.potential_kill_zones.append((id, np.array([kill_zone_x, kill_zone_y])))
                killed_ids.append(id)
            
        for id in killed_ids:
            self.last_other_drones_position.pop(id)

        for data in self.semantic_values():
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                drone_x = self.drone_position[0] + data.distance * math.cos(data.angle + self.get_angle())
                drone_y = self.drone_position[1] + data.distance * math.sin(data.angle + self.get_angle())
                self.semantic_drone_pos.append(np.array([drone_x, drone_y]))

        for id, potential_kill_zone in self.potential_kill_zones:
            count = 0
            for pos in self.semantic_drone_pos:
                if np.linalg.norm(pos - potential_kill_zone) < 10:
                    count += 1
            if count > 5:
                self.map.add_kill_zone(id, potential_kill_zone)
        print(len(self.semantic_drone_pos))
        if len(self.semantic_drone_pos) > 1000:
            self.semantic_drone_pos = self.semantic_drone_pos[-1000:]
    def control(self):
        if not self.odometer_values() is None:
            self.iteration += 1
            self.get_localization()
            self.found_center, self.command_semantic = self.process_semantic_sensor()
            self.process_communicator()
            self.check_wounded_available()
            if not self.communicator_is_disabled():
                self.check_other_drones_killed()
            else:
                self.last_other_drones_position = {}
                
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

            self.drone_repulsion()
            self.command["forward"] += self.repulsion[0]
            self.command["lateral"] += self.repulsion[1]
            self.command["forward"] = min(1,max(-1,self.command["forward"]))
            self.command["lateral"] = min(1,max(-1,self.command["lateral"]))

            self.drone_prev_position = self.drone_position.copy()
            
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
    
    # TODO : update confidence map using the velocity
    def update_mapping(self):
        """
        updates the map
        """
        self.estimated_pose = Pose(self.get_position(), self.get_angle())
        # max_vel_angle = 0.08
        # if abs(self.measured_angular_velocity()) < max_vel_angle:
        self.map.update(self.estimated_pose)

        # self.newmap.update_confidence_grid(self.estimated_pose, self.lidar())
        # self.newmap.update_occupancy_grid(self.estimated_pose, self.lidar())
        # self.newmap.update_map()
        # self.newmap.display_map()
        for other_drone in self.drone_list:
            if other_drone.id == self.identifier: continue
            self.map.merge(other_drone.map)


    def draw_top_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_repulsion:
            pos = self.get_position() + np.array(self.size_area)/2
            arcade.draw_line(pos[0], pos[1], pos[0]+self.repulsion[0]*20, pos[1]+self.repulsion[1]*20, arcade.color.PURPLE)

        if self.debug_lidar:
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
            if self.command["grasper"] == 1:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.RED)
            else:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.GREEN)

            direction = np.array([1,0])
            rot = np.array([[math.cos(self.get_anlge()), math.sin(self.get_anlge())],[-math.sin(self.get_anlge()), math.cos(self.get_anlge())]])
            direction = direction@rot
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*50, pos[1]+direction[1]*50, arcade.color.RED)

            direction = self.theorical_velocity
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*20, pos[1]+direction[1]*20, arcade.color.GREEN)


    def draw_bottom_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_path: 
            drawn_path = self.path.copy()
            if self.nextWaypoint is not None: drawn_path.append(self.nextWaypoint)
            drawn_path.append(self.get_position())
            for k in range(len(drawn_path)-1):
                pt1 = np.array(drawn_path[k]) + np.array(self.size_area)/2
                pt2 = np.array(drawn_path[k+1]) + np.array(self.size_area)/2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 255))

        if self.debug_kill_zones:
            for killed_drone_pos_grid in self.map.kill_zones.values():
                killed_drone_pos = self.map.grid_to_world(killed_drone_pos_grid)
                pos = np.array(killed_drone_pos) + np.array(self.size_area)/2
                # draw a rectangle
                arcade.draw_rectangle_filled(pos[0], pos[1], 100, 100, arcade.color.RED)
