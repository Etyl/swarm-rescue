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
from spg_overlay.utils.utils import circular_mean
from solutions.utils import normalize_angle, map_id_to_color
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.mapper.mapper import Map
from solutions.localizer.localizer import Localizer
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR
from solutions.types.types import DroneData
import solutions

POSITION_QUEUE_SIZE = 40 # number of positions to check if the drone is stuck
REFRESH_PATH_LIMIT = 40 # frames before refreshing the path

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
        self.drone_angle : float = 0 # The angle of the drone
        self.drone_angle_offset = 0 # The angle offset of the drone that can be changed by the states
        self.found_wounded = False # True if the drone has found a wounded person
        self.wounded_visible = False # True if the wounded person is visible
        self.found_center = False # True if the drone has found the rescue center
        self.last_angles = deque() # queue of the last angles
        self.last_positions = deque() # queue of the last positions
        self.repulsion = np.zeros(2) # The repulsion vector from the other drones
        self.wall_repulsion = np.zeros(2) # The repulsion vector from the walls
        self.center_angle = None
        self.is_near_center = False

        self.wounded_found_list = [] # the list of wounded persons found
        self.wounded_distance = 80 # The distance between wounded person to be considered as the same
        self.wounded_target = None # The wounded position to go to

        self.drone_list = [] # The list of drones

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = True
        self.debug_positions = False
        self.debug_map = False
        self.debug_roamer = False
        self.debug_controller = False 
        self.debug_lidar = False
        self.debug_repulsion = True
        self.debug_kill_zones = True
        self.debug_wall_repulsion = True
        self.debug_frontiers = False
        
        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        #self.map = Map(area_world=self.size_area, resolution=8, lidar=self.lidar(), debug_mode=self.debug_map)
        self.map = Map(drone=self, area_world=self.size_area, drone_lidar=self.lidar(), resolution=8, identifier=self.identifier, debug_mode=True)
        self.rescue_center_position = None
        
        self.roamer_controller = solutions.roamer.RoamerController(self, self.map, debug_mode=self.debug_roamer)

        self.localizer = Localizer()
        self.theorical_velocity = np.zeros(2)

        self.controller = solutions.drone_controller.DroneController(self, debug_mode=self.debug_controller)
        self.gps_disabled = True

        self.last_other_drones_position = {}
        self.kill_zones = []
        self.semantic_drone_pos = []
        self.potential_kill_zones = []
        self.kill_zone_mode = True

        self.point_of_interest = (0,0)
        self.frontiers = []
        self.selected_frontier_id = 0

        self.stuck_iteration = 0
        self.time = 0

    
    def compute_point_of_interest(self, number_of_drones: int):
        """
        computes the point of interest by repartitioning the drones in the area
        """
        size_rows = 450
        size_cols = 300
        num_rows = self.size_area[0] // size_rows
        num_cols = self.size_area[1] // size_cols


        num = max(num_rows, num_cols)

        size = max(size_rows, size_cols)

        drone_row = self.identifier % num
        drone_col = self.identifier // num

        return (drone_row * size + size/2, drone_col * size + size/2)


    def adapt_angle_direction(self, pos: list) -> float:
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
        if len(self.path) == 0: return dist < 30

        v1 = self.nextWaypoint - pos
        v2 = np.array(self.path[-1]) - np.array(self.nextWaypoint)

        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: 
            turning_angle = 0
        else:
            turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        return dist < 15 + (1+turning_angle)*20
        

    def define_message_for_all(self):
        data = DroneData()
        data.id = self.identifier
        data.position = self.get_position()
        data.angle = self.get_angle()
        data.vel_angle = self.compute_vel_angle()
        data.wounded_found = self.wounded_found_list
        data.wounded_target = self.wounded_target
        data.map = self.map
        data.semantic_values = self.semantic_values()
        data.kill_zone_mode = self.kill_zone_mode
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

        # check if wounded target is visible
        wounded_pos = get_wounded_position()
        if self.wounded_target is not None and np.linalg.norm(wounded_pos - self.wounded_target) < self.wounded_distance:
            self.wounded_visible = True
    
        for k in range(len(self.wounded_found_list)):
            wounded = self.wounded_found_list[k]
            if np.linalg.norm(wounded_pos - wounded["position"]) < self.wounded_distance:
                wounded["count"] += 1
                n = wounded["count"]
                wounded["position"] = wounded["position"]*((n-1)/n) + wounded_pos/n
                wounded["last_seen"] = 0
                return
            
        self.wounded_found_list.append({"position": wounded_pos, "count": 1, "last_seen": 0})
    
    def clear_wounded_found(self):
        frame_limit = 10
        
        for k in range(len(self.wounded_found_list)-1,-1,-1):
            self.wounded_found_list[k]["last_seen"] += 1
            if np.linalg.norm(self.get_position() - self.wounded_found_list[k]["position"])<self.wounded_distance/2 and self.wounded_found_list[k]["last_seen"] > frame_limit:
                self.wounded_found_list.pop(k)

    def update_drones(self, drone_data : DroneData):
        """
        updates the data of the drones
        """

        """
        # update the visibility of the drones
        for data in self.drone_list:
            data.visible = False
        #drone_data.visible = True
        """

        # update the wounded list
        if drone_data.wounded_target is not None:
            for k in range(len(self.wounded_found_list)):
                if np.linalg.norm(self.wounded_found_list[k]["position"] - drone_data.wounded_target) < self.wounded_distance:
                    #self.wounded_found_list[k]["taken"] = True
                    self.wounded_found_list[k]["drone_taker"] = drone_data.id
                    break
        
        # update the drone information
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
        
        self.found_wounded = False

        if self.wounded_found_list is None or len(self.wounded_found_list) == 0:
            return

        min_distance = np.inf
        best_position = None
        if (self.controller.current_state == self.controller.going_to_wounded 
            or self.controller.current_state == self.controller.approaching_wounded
            or self.controller.current_state == self.controller.roaming):
        
            # Select the best one among wounded persons detected
            for i,wounded in enumerate(self.wounded_found_list):
                distance = np.linalg.norm(self.get_position() - wounded["position"])

                # check if the wounded is taken by another drone
                if self.wounded_visible and "drone_taker" in wounded and wounded["drone_taker"] > self.identifier:
                    continue

                # check if the wounded is the target
                if "drone_taker" in wounded and wounded["drone_taker"] == self.identifier:
                    if self.controller.current_state != self.controller.roaming:
                        self.found_wounded = True
                        self.wounded_target = wounded["position"]
                        return
                    else:
                        wounded.pop("drone_taker")
                elif self.wounded_target is not None and np.linalg.norm(self.wounded_target - wounded["position"]) < 0.8*self.wounded_distance:
                    self.found_wounded = True
                    self.wounded_target = wounded["position"]
                    return
                
                if distance < min_distance:
                    min_distance = distance
                    best_position = i

        if best_position is None:
            return
        
        if self.controller.current_state == self.controller.approaching_wounded:
            self.wounded_found_list[i]["drone_taker"] = self.identifier
        
        self.found_wounded = True
        self.wounded_target = self.wounded_found_list[i]["position"]


    def get_control_from_semantic(self):

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

        angular_vel_controller_max = 1.0

        command = {"forward": 1.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0}

        if self.found_wounded:
            best_angle = normalize_angle(angle(np.array([1,0]), self.wounded_target - self.drone_position))
            best_angle = normalize_angle(best_angle - normalize_angle(self.get_angle()))

            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.4        

        if self.found_center:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * self.center_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if self.found_center and self.is_near_center:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

        detection_semantic = self.semantic_values()
        
        self.clear_wounded_found()

        self.wounded_visible = False
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
            self.center_angle = circular_mean(np.array(angles_list))
            self.is_near_center = is_near
        else:
            self.center_angle = None
            self.is_near_center = False

        return found_rescue_center


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


    def update_drone_repulsion(self):
        """
        repulses the drone from other drones
        """
        def compute_angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        def repulsion_dist(dist):
            a = 7
            b = 0.8
            c = 0.007
            if dist <= a: return 2
            return max(0,min(2,MAX_RANGE_SEMANTIC_SENSOR*(b/(dist-a) - c)))
        
        repulsion = np.zeros(2)
        min_dist = np.inf
        for data in self.semantic_values():
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                angle = data.angle
                dist = data.distance
                min_dist = min(min_dist, dist)
                repulsion += repulsion_dist(dist) * np.array([math.cos(angle), math.sin(angle)])
        
        if np.linalg.norm(repulsion) == 0:
            self.repulsion = np.zeros(2)
            return
        
        repulsion = -repulsion/np.linalg.norm(repulsion)
        repulsion *= repulsion_dist(min_dist)

        if (self.controller.current_state == self.controller.going_to_center or
            self.controller.current_state == self.controller.approaching_wounded or
            self.controller.current_state == self.controller.approaching_center):
            self.repulsion = 0.05*repulsion
        else:
            self.repulsion = repulsion
      

    def update_wall_repulsion(self):
        """
        update repulsion vector for the drone from the walls (local drone vector)
        """
        def compute_angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

        lidar_dist = self.lidar().get_sensor_values()
        lidar_angles = self.lidar().ray_angles
        detection_semantic = self.semantic_values()

        drone_angles = []
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                drone_angles.append(round((data.angle+np.pi)*34/(2*np.pi)))
        
        repulsion_vectors = [np.zeros(2)]

        if self.controller.current_state == self.controller.going_to_center:
            for k in range(35):
                i = round(k*180/34)
                if (k not in drone_angles and
                    lidar_dist[i] < 0.3 * MAX_RANGE_LIDAR_SENSOR and 
                    abs(lidar_angles[i])>np.pi/3):
                        d = 1 - lidar_dist[i]/MAX_RANGE_LIDAR_SENSOR
                        repulsion_vectors.append(np.array([d*math.cos(lidar_angles[i]), d*math.sin(lidar_angles[i])]))
        else:
            for k in range(35):
                i = round(k*180/34)
                if (k not in drone_angles and
                    lidar_dist[i] < 0.3 * MAX_RANGE_LIDAR_SENSOR):
                        d = 1 - lidar_dist[i]/MAX_RANGE_LIDAR_SENSOR
                        repulsion_vectors.append(np.array([d*math.cos(lidar_angles[i]), d*math.sin(lidar_angles[i])])) 

        # check if drone is too close to a wall
        kmin = np.argmax(np.linalg.norm(repulsion_vectors, axis=1))
        if np.linalg.norm(repulsion_vectors[kmin]) >= 0.93:
            self.wall_repulsion = -repulsion_vectors[kmin]/np.linalg.norm(repulsion_vectors[kmin])
            return

        repulsion_vector = -sum(repulsion_vectors)

        if np.linalg.norm(repulsion_vector) == 0:
            self.wall_repulsion = np.zeros(2)
            return

        # check if the repulsion vector is needed (the repulsion vector repulses from an open space)
        repulsion_angle = compute_angle(np.array([1,0]), -repulsion_vector/np.linalg.norm(repulsion_vector))
        kmin = np.argmin(np.abs(lidar_angles - repulsion_angle))

        if (lidar_dist[kmin] >= 0.25*MAX_RANGE_LIDAR_SENSOR):
            self.wall_repulsion = np.zeros(2)
            return

        coef = 0
        if lidar_dist[kmin] < 50:
            coef = max(0,2 * (1 - 2*(min(lidar_dist)-11)/(MAX_RANGE_LIDAR_SENSOR)))
        
        
        repulsion_vector = repulsion_vector/np.linalg.norm(repulsion_vector)
        self.wall_repulsion = coef * repulsion_vector

        # TODO : change repulsion according to drone direction (change forward and sideways command)

    def test_stuck(self):
        self.last_positions.append(self.get_position())
        if len(self.last_positions) >= POSITION_QUEUE_SIZE:
            self.last_positions.popleft()
        else:
            return
        for k in range(len(self.last_positions)):
            if np.linalg.norm(self.last_positions[0] - self.last_positions[k]) > 10:
                return
        if self.stuck_iteration < REFRESH_PATH_LIMIT:
            return
        self.stuck_iteration = 0
        self.path = []
        self.nextWaypoint = None
        if self.controller.current_state == self.controller.approaching_wounded:
            self.controller.force_drone_stuck()


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
        for drone in self.drone_list:
            if drone.id == self.identifier: continue
            self.last_other_drones_position[drone.id] = [drone.position, drone.vel_angle]
        
        # check if other drones are killed by checking it's not in drone_list anymore and have last seen distance < MAX_RANGE_LIDAR_SENSOR / 2
        killed_ids = []
        for id in self.last_other_drones_position:
            if id not in [drone.id for drone in self.drone_list]:
                if np.linalg.norm(self.last_other_drones_position[id][0] - self.drone_position) < RANGE_COMMUNICATION * 0.85:
                    #print(f"Drone {id} killed")
                    self.path = []
                    self.nextWaypoint = None
                    kill_zone_x = self.last_other_drones_position[id][0][0] + 50*math.cos(self.last_other_drones_position[id][1])
                    kill_zone_y = self.last_other_drones_position[id][0][1] + 50*math.sin(self.last_other_drones_position[id][1])
                    self.map.add_kill_zone(id, [kill_zone_x, kill_zone_y])
                    killed_ids.append(id)
                else:
                    #print(f"Drone {id} left")
                    killed_ids.append(id)
            
        for id in killed_ids:
            self.last_other_drones_position.pop(id)
        
        for id, kill_zone in self.map.kill_zones.items():
            if id in [drone.id for drone in self.drone_list]:
                self.kill_zone_mode = False
                #print("Kill zone mode disabled")

        # killed_ids = []
        # for id, kill_zone in self.map.kill_zones.items():
        #     # check if the kill zone is near the drone
        #     if np.linalg.norm(np.array(kill_zone) - self.drone_position) < MAX_RANGE_SEMANTIC_SENSOR:
        #         real_kill_zone = False
        #         for data in self.semantic_values():
        #             if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
        #                 drone_x = self.drone_position[0] + data.distance * math.cos(data.angle + self.get_angle())
        #                 drone_y = self.drone_position[1] + data.distance * math.sin(data.angle + self.get_angle())
        #                 if np.linalg.norm(np.array(kill_zone) - np.array([drone_x, drone_y])) < 100:
        #                     real_kill_zone = True
        #         if not real_kill_zone:
        #             killed_ids.append(id)
        #             print("Not a real kill zone")
                    
        # for id in killed_ids:
        #     self.map.remove_kill_zone(id)
                

    def control(self):

        # check if drone is dead
        if self.odometer_values() is None:
            return
        
        self.stuck_iteration += 1
        self.time += 1

        self.get_localization()
        self.found_center = self.process_semantic_sensor()
        self.process_communicator()
        self.check_wounded_available()
        if not self.communicator_is_disabled():
            self.check_other_drones_killed()
        self.test_stuck()
        
        if self.rescue_center_position is None:
            self.compute_rescue_center_position()
        
        for drone in self.drone_list:
            if drone.id == self.identifier: continue
            if not drone.kill_zone_mode:
                self.kill_zone_mode = False
        if not self.communicator_is_disabled() and self.stuck_iteration > 2 and self.kill_zone_mode:
            self.check_other_drones_killed()
        else:
            self.last_other_drones_position = {}

        if self.time > 20:
            if self.roaming:
                try:
                    self.roamer_controller.cycle()
                except exceptions.TransitionNotAllowed:
                    pass
            
        self.controller.cycle()
        self.roaming = self.controller.current_state == self.controller.roaming
        
        self.update_mapping()
            
        if self.roaming:
            self.command = self.roamer_controller.command.copy()
        else:
            self.command = self.controller.command.copy()

        if self.gps_disabled:
            self.command["rotation"] /=2
            self.command["forward"] /=2
            self.command["lateral"] /=2

        self.update_drone_repulsion()
        self.update_wall_repulsion()
        self.command["forward"] += self.repulsion[0]
        self.command["lateral"] += self.repulsion[1]
        if (self.controller.current_state == self.controller.going_to_center):
            self.command["forward"] += 0.9*self.wall_repulsion[0]
            self.command["lateral"] += 0.9*self.wall_repulsion[1]
        elif (self.controller.current_state != self.controller.approaching_wounded 
            and self.controller.current_state != self.controller.approaching_center) :
            self.command["forward"] += self.wall_repulsion[0]
            self.command["lateral"] += self.wall_repulsion[1]
        
        self.command["forward"] = min(1,max(-1,self.command["forward"]))
        self.command["lateral"] = min(1,max(-1,self.command["lateral"]))

        self.drone_prev_position = self.drone_position.copy()

        self.point_of_interest = self.compute_point_of_interest(10)
            
        return self.command
    
    
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


    def set_selected_frontier_id(self, id):
        """
        sets the selected frontier id
        """
        self.selected_frontier_id = id

    def draw_top_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_repulsion:
            pos = self.get_position() + np.array(self.size_area)/2
            angle = self.get_angle()
            rot = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
            repulsion = self.repulsion @ rot
            arcade.draw_line(pos[0], pos[1], pos[0]+repulsion[0]*20, pos[1]+repulsion[1]*20, arcade.color.PURPLE)

        if self.debug_wall_repulsion:
            pos = self.get_position() + np.array(self.size_area)/2
            angle = self.get_angle()
            rot = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
            wall_repulsion = -self.wall_repulsion @ rot
            arcade.draw_line(pos[0], pos[1], pos[0]+wall_repulsion[0]*0.25*MAX_RANGE_LIDAR_SENSOR, pos[1]+wall_repulsion[1]*0.25*MAX_RANGE_LIDAR_SENSOR, arcade.color.RED)

        if self.debug_lidar:
            lidar_dist = self.lidar().get_sensor_values()[::].copy()
            lidar_angles = self.lidar().ray_angles[::].copy()
            for k in range(len(lidar_dist)):
                pos = self.get_position() + np.array(self.size_area)/2
                pos[0] += lidar_dist[k]*math.cos(lidar_angles[k]+self.get_angle())
                pos[1] += lidar_dist[k]*math.sin(lidar_angles[k]+self.get_angle())
                arcade.draw_circle_filled(pos[0], pos[1],2, arcade.color.PURPLE)

        if self.debug_wounded:
            for wounded in self.wounded_found_list:
                pos = np.array(wounded["position"]) + np.array(self.size_area)/2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.GREEN_YELLOW)
                arcade.draw_circle_outline(pos[0], pos[1],self.wounded_distance, arcade.color.GREEN_YELLOW)
            if self.wounded_target is not None and (self.controller.current_state == self.controller.going_to_wounded or self.controller.current_state == self.controller.approaching_wounded):
                pos = np.array(self.wounded_target) + np.array(self.size_area)/2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.RED)
                arcade.draw_circle_outline(pos[0], pos[1],self.wounded_distance, arcade.color.RED)

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

        # draw frontiers
        if self.debug_frontiers:
            pos = np.array(self.get_position()) + np.array(self.size_area)/2
            arcade.draw_text(str(self.identifier), pos[0], pos[1], map_id_to_color[self.identifier], 20)
            for id, frontier in enumerate(self.frontiers):
                for point in frontier:
                    if id == self.selected_frontier_id:
                        pos = np.array(self.map.grid_to_world(point)) + np.array(self.size_area)/2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 8, 8, map_id_to_color[self.identifier])
                    else:
                        pos = np.array(self.map.grid_to_world(point)) + np.array(self.size_area)/2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 2, 2, map_id_to_color[self.identifier])

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

        if self.debug_kill_zones and self.kill_zone_mode:
            for killed_drone_pos in self.map.kill_zones.values():
                pos = np.array(killed_drone_pos) + np.array(self.size_area)/2
                # draw a rectangle
                arcade.draw_rectangle_filled(pos[0], pos[1], 100, 100, arcade.color.RED)
