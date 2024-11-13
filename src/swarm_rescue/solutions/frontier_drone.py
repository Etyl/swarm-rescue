from __future__ import annotations

import math
import numpy as np
from typing import Optional, List, Tuple, Deque, Dict
import arcade
from collections import deque

from statemachine import exceptions

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean
from solutions.utils.utils import normalize_angle, map_id_to_color
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR

from solutions.utils.types import DroneData, WoundedData, Vector2D
from solutions.localizer.localizer import Localizer
from solutions.mapper.mapper import Map
from solutions.drone_controller import DroneController
from solutions.roamer.roamer import RoamerController

class FrontierDrone(DroneAbstract):

    POSITION_QUEUE_SIZE = 30 # number of positions to check if the drone is stuck
    REFRESH_PATH_LIMIT = 40 # frames before refreshing the path
    WOUNDED_DISTANCE = 50 # The distance between wounded person to be considered as the same

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        **kwargs):

        super().__init__(
            identifier=identifier,
            misc_data=misc_data,
            display_lidar_graph=False,
            **kwargs)

        self.path : List[Vector2D] = []
        self.waypoint_index : Optional[int] = None # The index to the next waypoint to go to
        self.drone_angle_offset : float = 0 # The angle offset of the drone that can be changed by the states
        self.found_wounded : bool = False # True if the drone has found a wounded person
        self.last_positions : Deque[Vector2D] = deque()
        self.last_wounded_seen : int = 10000 # time (frames) since the wounded was last seen
        self.repulsion : Vector2D = Vector2D(0,0) # The repulsion vector from the other drones
        self.wall_repulsion : Vector2D = Vector2D(0,0) # The repulsion vector from the walls
        self.center_angle : Optional[float] = None # angle from the visible rescue center
        self.rescue_center_dist : Optional[float] = None # distance from the visible rescue center
        self.ignore_repulsion : int = 0 # timer to ignore the repulsion vector (>0 => ignore)
        self.target: Optional[Vector2D] = None # target for the drone for the path planning
        self.killed_drones : List[int] = [] # the list of killed drones ids
        self.prev_command : Dict = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }
        self.visible_drones: List[Tuple[int,Vector2D]] = [] # List of (id,drone_position)

        self.rescue_center_position: Optional[Vector2D] = None
        self.wounded_found_list : List[WoundedData] = [] # the list of wounded persons found
        self.wounded_target : Optional[Vector2D] = None # The wounded position to go to
        self.drone_list : List[DroneData] = [] # The list of drones

        ## Debug controls

        self.debug_path = False # True if the path must be displayed
        self.debug_wounded = False
        self.debug_positions = True
        self.debug_map = False
        self.debug_roamer = False
        self.debug_controller = False
        self.debug_mapper = False
        self.debug_lidar = False
        self.debug_repulsion = False
        self.debug_kill_zones = True
        self.debug_wall_repulsion = False
        self.debug_frontiers = False

        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self.command: Dict[str, float] = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }
        self.prev_command: Dict[str, float] = self.command

        self.map : Map = Map(area_world=self.size_area, resolution=4, debug_mode=self.debug_mapper)
        self.roamer_controller : RoamerController = RoamerController(self, self.map, debug_mode=self.debug_roamer)

        self.localizer : Localizer = Localizer(self)

        self.controller : DroneController = DroneController(self, debug_mode=self.debug_controller)

        self.last_other_drones_position : Dict[int, Tuple[Vector2D, float]] = {}
        self.kill_zones : List[Vector2D] = []
        self.potential_kill_zones : List[Vector2D] = []
        self.kill_zone_mode = True

        self.point_of_interest = (0,0)
        self.frontiers : List[List[Vector2D]] = []
        self.selected_frontier_id : int = 0

        self.stuck_iteration : int = 0
        self.time : int = 0
        self.time_in_no_gps : int = 0
        self.previous_drone_health: int = self.drone_health

    @property
    def near_center(self) -> bool:
        return self.rescue_center_dist is not None and self.rescue_center_dist < MAX_RANGE_SEMANTIC_SENSOR*(2/3)

    @property
    def gps_disabled(self) -> bool:
        return self.measured_compass_angle() is None

    @property
    def found_center(self) -> bool:
        return self.rescue_center_position is not None

    @property
    def wounded_visible(self) -> bool:
        return self.last_wounded_seen < 20

    @property
    def next_waypoint(self) -> Optional[Vector2D]:
        if self.waypoint_index is None or self.waypoint_index>=len(self.path):
            return None
        else:
            return self.path[self.waypoint_index]

    @property
    def has_collided(self) -> bool:
        return self.previous_drone_health != self.drone_health

    @property
    def drone_position(self) -> Vector2D:
        return self.localizer.drone_position

    @property
    def drone_angle(self) -> float:
        return self.localizer.drone_angle


    def compute_point_of_interest(self):
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

        return drone_row * size + size/2, drone_col * size + size/2


    def check_waypoint(self):
        """
        checks if the drone has reached the waypoint
        """
        next_waypoint: Vector2D = self.next_waypoint
        if next_waypoint is None:
            return False

        return self.drone_position.distance(next_waypoint) < 40


    def define_message_for_all(self):
        data = DroneData(
            id = self.identifier,
            position = self.get_position(),
            angle = self.get_angle(),
            vel_angle = self.localizer.drone_velocity_angle,
            wounded_target = self.wounded_target,
            map = self.map,
            semantic_values = self.semantic_values(),
            kill_zone_mode = self.kill_zone_mode,
            next_waypoint= self.next_waypoint,
            killed_drones = self.killed_drones,
            confidence_position = self.localizer.confidence_position,
            visible_drones = self.visible_drones
        )
        return data

    def get_position(self) -> Vector2D:
        """
        returns the position of the drone
        """
        return self.drone_position

    def get_angle(self) -> float:
        """
        returns the angle of the drone
        """
        return self.drone_angle


    def add_wounded(self, data_wounded):
        """
        compares the wounded persons detected with the ones already detected
        """

        def get_wounded_position():
            pos = self.drone_position.copy()
            angle = normalize_angle(self.get_angle() + data_wounded.angle)
            pos.setX(pos.x + data_wounded.distance * math.cos(angle))
            pos.setY(pos.y + data_wounded.distance * math.sin(angle))
            return pos

        # check if wounded target is visible
        wounded_pos = get_wounded_position()
        if self.wounded_target is not None and wounded_pos.distance(self.wounded_target) < FrontierDrone.WOUNDED_DISTANCE:
            self.last_wounded_seen = 0

        for k in range(len(self.wounded_found_list)):
            wounded = self.wounded_found_list[k]
            if wounded_pos.distance(wounded.position) < FrontierDrone.WOUNDED_DISTANCE:
                wounded.count += 1
                n = wounded.count
                wounded.position = wounded.position*((n-1)/n) + wounded_pos/n
                wounded.last_seen = 0
                return

        self.wounded_found_list.append(WoundedData(wounded_pos, 1, 0, None))


    def clear_wounded_found(self):

        for k in range(len(self.wounded_found_list)-1,-1,-1):
            self.wounded_found_list[k].last_seen += 1
            if self.get_position().distance(self.wounded_found_list[k].position) < FrontierDrone.WOUNDED_DISTANCE and self.wounded_found_list[k].last_seen > 5:
                self.wounded_found_list.pop(k)

    def update_drones(self, drone_data : DroneData):
        """
        updates the data of the drones
        """
        # update the wounded list
        if drone_data.wounded_target is not None:
            for k in range(len(self.wounded_found_list)):
                if self.wounded_found_list[k].position.distance(drone_data.wounded_target) < FrontierDrone.WOUNDED_DISTANCE:
                    #self.wounded_found_list[k].taken = True
                    self.wounded_found_list[k].drone_taker = drone_data.id
                    break

        # update the kill zones
        if not drone_data.kill_zone_mode and self.kill_zone_mode:
            print("Found someone in no kill zone mode")
            self.reset_kill_zones()

        # update killed drones
        for killed in drone_data.killed_drones:
            if killed not in self.killed_drones:
                self.killed_drones.append(killed)

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
                distance = self.get_position().distance(wounded.position)

                # check if the wounded is taken by another drone
                if self.wounded_visible and wounded.drone_taker is not None and wounded.drone_taker > self.identifier:
                    continue

                # check if the wounded is the target
                if wounded.drone_taker is not None and wounded.drone_taker == self.identifier:
                    if (self.controller.current_state == self.controller.approaching_wounded or
                        self.controller.current_state == self.controller.going_to_wounded):
                        self.found_wounded = True
                        self.wounded_target = wounded.position
                        return
                    else:
                        wounded.drone_taker = None
                elif self.wounded_target is not None and self.wounded_target.distance(wounded.position) < 0.8*FrontierDrone.WOUNDED_DISTANCE:
                    self.found_wounded = True
                    self.wounded_target = wounded.position
                    return

                if distance < min_distance:
                    min_distance = distance
                    best_position = i

        if best_position is None:
            return

        if self.controller.current_state == self.controller.approaching_wounded:
            self.wounded_found_list[best_position].drone_taker = self.identifier

        self.found_wounded = True
        self.wounded_target = self.wounded_found_list[best_position].position


    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        detection_semantic = self.semantic_values()

        self.clear_wounded_found()
        self.visible_drones = []

        self.last_wounded_seen += 1
        if detection_semantic:
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    # wounded: WoundedData = WoundedData(data)
                    self.add_wounded(data)

                if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                    drone_pos = self.drone_position.copy()
                    angle = data.angle + self.drone_angle
                    drone_pos += data.distance * Vector2D(math.cos(angle), math.sin(angle))
                    best_d,identifier = 10000000,-1
                    for drone in self.drone_list:
                        d = drone_pos.distance(drone.position)
                        if d < best_d:
                            best_d,identifier = d,drone.id
                    if best_d < 50:
                        self.visible_drones.append((identifier,drone_pos))

        rescue_center_dist = float('inf')
        angles_list = []
        if (detection_semantic and
            (self.controller.current_state == self.controller.going_to_center
            or self.controller.current_state == self.controller.approaching_center)):
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    rescue_center_dist = min(rescue_center_dist,data.distance)
                    angles_list.append(data.angle)


        if rescue_center_dist != float('inf'):
            self.center_angle = circular_mean(np.array(angles_list))
            self.rescue_center_dist = rescue_center_dist
        else:
            self.center_angle = None
            self.rescue_center_dist = None


    def get_path(self, pos:Vector2D):
        """
        returns the path to the destination
        """
        return self.map.shortest_path(self.drone_position, pos)


    # TODO improve using pure pursuit
    def update_waypoint_index(self) -> None:
        if self.waypoint_index is None:
            self.target = None
            return

        curr_proj: Optional[Vector2D] = None
        if 0 < self.waypoint_index <= len(self.path) - 1 and self.path[self.waypoint_index-1] != self.path[self.waypoint_index]:
            curr_proj = self.get_position().project(self.path[self.waypoint_index-1],self.path[self.waypoint_index])
            adv_proj = curr_proj + (self.path[self.waypoint_index]-self.path[self.waypoint_index-1]).normalize()*40
            if adv_proj.distance(self.path[self.waypoint_index-1]) >= self.path[self.waypoint_index].distance(self.path[self.waypoint_index-1]):
                curr_proj = self.path[self.waypoint_index]
            else:
                curr_proj = adv_proj

        self.target = curr_proj


    def compute_rescue_center_position(self):
        """
        computes the position of the rescue center
        """
        semantic_lidar = [data for data in self.semantic_values() if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER]

        if len(semantic_lidar)>0 and self.rescue_center_position is None:
            self.rescue_center_position = self.drone_position.copy()


    def update_drone_repulsion(self):
        """
        repulses the drone from other drones
        """
        def repulsion_dist(distance):
            a = 7
            b = 0.8
            c = 0.007
            if distance <= a: return 2
            return max(0,min(2,MAX_RANGE_SEMANTIC_SENSOR*(b/(distance-a) - c)))

        repulsion = Vector2D(0,0)
        min_dist = np.inf
        for data in self.semantic_values():
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                angle = data.angle
                dist = data.distance
                min_dist = min(min_dist, dist)
                drone = None
                for d in self.drone_list:
                    if d.position.distance(Vector2D(dist*math.cos(angle+self.drone_angle), dist*math.sin(angle+self.drone_angle))+self.get_position()) < 30:
                        drone = d
                if drone is None or drone.id>self.identifier:
                    repulsion += Vector2D(math.cos(angle), math.sin(angle)) * repulsion_dist(dist)
                else:
                    repulsion += 0.2*Vector2D(math.cos(angle), math.sin(angle)) * repulsion_dist(dist)

        if repulsion.norm() == 0:
            self.repulsion = Vector2D(0,0)
            return

        repulsion = -repulsion.normalize()
        repulsion *= repulsion_dist(min_dist)

        if (self.controller.current_state == self.controller.going_to_center or
            self.controller.current_state == self.controller.approaching_wounded or
            self.controller.current_state == self.controller.approaching_center):
            self.repulsion = 0.05*repulsion
        else:
            self.repulsion = repulsion


    def update_wall_repulsion(self) -> None:
        """
        update repulsion vector for the drone from the walls (local drone vector)
        """

        lidar_dist = self.lidar().get_sensor_values()
        lidar_angles = self.lidar().ray_angles
        detection_semantic = self.semantic_values()

        drone_angles = []
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.DRONE:
                drone_angles.append(round((data.angle+np.pi)*34/(2*np.pi)))

        repulsion_vectors : List[Vector2D] = [Vector2D(0,0)]

        if self.controller.current_state == self.controller.going_to_center:
            for k in range(35):
                i = round(k*180/34)
                if (k not in drone_angles and
                    lidar_dist[i] < 0.3 * MAX_RANGE_LIDAR_SENSOR and
                    abs(lidar_angles[i])>np.pi/3):
                        d = 1 - lidar_dist[i]/MAX_RANGE_LIDAR_SENSOR
                        repulsion_vectors.append(Vector2D(d*math.cos(lidar_angles[i]), d*math.sin(lidar_angles[i])))
        else:
            for k in range(35):
                i = round(k*180/34)
                if (k not in drone_angles and
                    lidar_dist[i] < 0.3 * MAX_RANGE_LIDAR_SENSOR):
                        d = 1 - lidar_dist[i]/MAX_RANGE_LIDAR_SENSOR
                        repulsion_vectors.append(Vector2D(d*math.cos(lidar_angles[i]), d*math.sin(lidar_angles[i])))

        # check if drone is too close to a wall
        kmin = np.argmax([v.norm() for v in repulsion_vectors])
        if repulsion_vectors[kmin].norm() >= 0.94:
            self.wall_repulsion = repulsion_vectors[kmin].normalize() * -1.5
            return

        repulsion_vector = Vector2D(0,0)
        for v in repulsion_vectors:
            repulsion_vector -= v

        if repulsion_vector.norm() == 0:
            self.wall_repulsion = Vector2D(0,0)
            return

        # check if the repulsion vector is needed (the repulsion vector repulses from an open space)
        repulsion_angle = Vector2D(1,0).get_angle(-repulsion_vector)
        kmin = np.argmin(np.abs(lidar_angles - repulsion_angle))

        if lidar_dist[kmin] >= 0.25*MAX_RANGE_LIDAR_SENSOR:
            self.wall_repulsion = Vector2D(0,0)
            return

        coef = 0
        if lidar_dist[kmin] < 40:
            coef = min(2, max(0, 2 * (1 - 3 * (min(lidar_dist)-13) / MAX_RANGE_LIDAR_SENSOR)))

        repulsion_vector = repulsion_vector.normalize()
        self.wall_repulsion = repulsion_vector * coef

        # TODO : change repulsion according to drone direction (change forward and sideways command)

    def test_stuck(self):
        self.last_positions.append(self.get_position())
        if len(self.last_positions) >= FrontierDrone.POSITION_QUEUE_SIZE:
            self.last_positions.popleft()
        else:
            return
        for k in range(len(self.last_positions)):
            if self.last_positions[0].distance(self.last_positions[k]) > 10:
                return
        if self.stuck_iteration < FrontierDrone.REFRESH_PATH_LIMIT:
            return
        self.stuck_iteration = 0

        closest_drone = None
        dist = np.inf
        for drone in self.drone_list:
            if closest_drone is None or drone.position.distance(self.get_position()) < dist:
                dist = drone.position.distance(self.get_position())
                closest_drone = drone

        if (closest_drone is not None and
            self.next_waypoint is not None and
            closest_drone.next_waypoint is not None and
            self.next_waypoint.distance(closest_drone.next_waypoint) < 30 and
            self.identifier < closest_drone.id):

            self.ignore_repulsion = 30

        self.reset_path()
        if self.controller.current_state == self.controller.approaching_wounded:
            self.controller.force_drone_stuck()

    
    def update_last_other_drones_position(self):
        """
        updates the last drones seen
        """
        self.last_other_drones_position = {}
        for drone in self.drone_list:
            if drone.id == self.identifier: continue
            self.last_other_drones_position[drone.id] = (drone.position, drone.vel_angle)

    def reset_path(self) -> None:
        """
        resets the path
        """
        self.path = []
        self.waypoint_index = None

    def set_path(self, path: Optional[List[Vector2D]]) -> None:
        """
        set a new path
        """
        if path is None:
            return
        self.path = path
        self.waypoint_index = 0

    def compute_kill_zone(self):
        """
        computes the kill zone
        """
        other_drones_id = [drone.id for drone in self.drone_list]
        killed_ids = []
        for droneId in self.last_other_drones_position:
            if droneId not in other_drones_id:
                last_pos = self.last_other_drones_position[droneId][0]
                last_angle = self.last_other_drones_position[droneId][1]
                if (last_pos.distance(self.drone_position)) < RANGE_COMMUNICATION * 0.7:
                    kill_zone_x = last_pos.x + 30 * math.cos(last_angle)
                    kill_zone_y = last_pos.y + 30 * math.sin(last_angle)
                    self.map.add_kill_zone(droneId, Vector2D(kill_zone_x, kill_zone_y))
                    self.reset_path()
                    killed_ids.append(droneId)
        for droneId in killed_ids:
            self.last_other_drones_position.pop(droneId)
            self.killed_drones.append(droneId)

    def check_if_no_com_zone_mode(self):
        """
        checks if the map is in no communication zone mode by checking if we communicate with other drones that should be dead
        """
        for drone in self.drone_list:
            if drone.id == self.identifier: continue
            if drone.id in self.killed_drones:
                print("No communication zone mode")
                self.reset_kill_zones()

    def reset_kill_zones(self):
        print("No communication zone mode")
        self.kill_zones = []
        self.killed_drones = []
        self.map.reset_kill_zones()
        self.kill_zone_mode = False
        self.reset_path()

    def control(self):

        # check if drone is dead
        if self.odometer_values() is None:
            return
        
        self.stuck_iteration += 1
        self.time += 1
        if self.gps_disabled:
            self.time_in_no_gps += 1

        self.localizer.localize()
        self.process_semantic_sensor()
        self.process_communicator()
        self.check_wounded_available()
        self.test_stuck()

        if self.rescue_center_position is None:
            self.compute_rescue_center_position()

        if self.kill_zone_mode:
            self.compute_kill_zone()
            self.check_if_no_com_zone_mode()
        self.update_last_other_drones_position()
        if self.time > 50:
            if self.roaming:
                try:
                    self.roamer_controller.cycle()
                except exceptions.TransitionNotAllowed:
                    pass

        self.controller.cycle()

        self.roaming = self.controller.current_state == self.controller.roaming

        self.update_mapping()

        if self.roaming and self.time > 50:
            self.command = self.roamer_controller.command.copy()
        else:
            self.command = self.controller.command.copy()

        if self.gps_disabled:
            self.command["rotation"] /=2
            self.command["forward"] /=2
            self.command["lateral"] /=2

        self.update_drone_repulsion()
        self.update_wall_repulsion()

        if self.ignore_repulsion <= 0:
            self.command["forward"] += self.repulsion.x
            self.command["lateral"] += self.repulsion.y
        else:
            self.ignore_repulsion -= 1

        if self.controller.current_state == self.controller.going_to_center:
            self.command["forward"] += 0.9*self.wall_repulsion.x
            self.command["lateral"] += 0.9*self.wall_repulsion.y
        elif (self.controller.current_state != self.controller.approaching_wounded
            and self.controller.current_state != self.controller.approaching_center) :
            self.command["forward"] += self.wall_repulsion.x
            self.command["lateral"] += self.wall_repulsion.y

        self.command["forward"] = min(1,max(-1,self.command["forward"]))
        self.command["lateral"] = min(1,max(-1,self.command["lateral"]))

        self.point_of_interest = self.compute_point_of_interest()
        self.prev_command = self.command
        self.previous_drone_health = self.drone_health

        return self.command


    # TODO : update confidence map using the velocity
    def update_mapping(self):
        """
        updates the map
        """
        estimated_pose = Pose(self.get_position().array, self.get_angle())
        # max_vel_angle = 0.08
        # if abs(self.measured_angular_velocity()) < max_vel_angle:
        if not self.gps_disabled:
            self.map.update(estimated_pose, self.lidar(), self)
        else:
            max_vel_angle = 0.08
            if abs(self.measured_angular_velocity()) < max_vel_angle:
                self.map.update_confidence_wall_map()
                self.map.update(estimated_pose, self.lidar(), self)
        # self.newmap.update_confidence_grid(self.estimated_pose, self.lidar())
        # self.newmap.update_occupancy_grid(self.estimated_pose, self.lidar())
        # self.newmap.update_map()
        # self.newmap.display_map()
        for other_drone in self.drone_list:
            if other_drone.id == self.identifier: continue
            self.map.merge(other_drone.map, self)


    def set_selected_frontier_id(self, frontier_id : int):
        """
        sets the selected frontier id
        """
        self.selected_frontier_id = frontier_id

    def draw_top_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_repulsion:
            pos = self.get_position().array + np.array(self.size_area) / 2
            repulsion = self.repulsion.copy()
            repulsion.rotate(self.get_angle())
            arcade.draw_line(pos[0], pos[1], pos[0]+repulsion.x*20, pos[1]+repulsion.y*20, arcade.color.PURPLE)

        if self.debug_wall_repulsion:
            pos = self.get_position().array + np.array(self.size_area) / 2
            wall_repulsion = self.wall_repulsion.copy()
            wall_repulsion.rotate(self.get_angle())
            arcade.draw_line(pos[0], pos[1], pos[0]+wall_repulsion.x*0.25*MAX_RANGE_LIDAR_SENSOR, pos[1]+wall_repulsion.y*0.25*MAX_RANGE_LIDAR_SENSOR, arcade.color.RED)

        if self.debug_lidar:
            lidar_dist = self.lidar().get_sensor_values()[::].copy()
            lidar_angles = self.lidar().ray_angles[::].copy()
            for k in range(len(lidar_dist)):
                pos = self.get_position().array + np.array(self.size_area) / 2
                pos[0] += lidar_dist[k]*math.cos(lidar_angles[k]+self.get_angle())
                pos[1] += lidar_dist[k]*math.sin(lidar_angles[k]+self.get_angle())
                arcade.draw_circle_filled(pos[0], pos[1],2, arcade.color.PURPLE)

        if self.debug_wounded:
            for wounded in self.wounded_found_list:
                pos = wounded.position.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.GREEN_YELLOW)
                arcade.draw_circle_outline(pos[0], pos[1],FrontierDrone.WOUNDED_DISTANCE, arcade.color.GREEN_YELLOW)
            if self.wounded_target is not None and (self.controller.current_state == self.controller.going_to_wounded or self.controller.current_state == self.controller.approaching_wounded):
                pos = self.wounded_target.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.RED)
                arcade.draw_circle_outline(pos[0], pos[1],FrontierDrone.WOUNDED_DISTANCE, arcade.color.RED)

        if self.debug_positions:
            pos = self.get_position().array + np.array(self.size_area) / 2
            if self.command["grasper"] == 1:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.RED)
            else:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.GREEN)

            direction = np.array([1,0])
            rot = np.array([[math.cos(self.get_angle()), math.sin(self.get_angle())],[-math.sin(self.get_angle()), math.cos(self.get_angle())]])
            direction = direction@rot
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*50, pos[1]+direction[1]*50, arcade.color.RED)
            arcade.draw_text(f"{self.localizer.confidence_position:.3f}", pos[0]+10, pos[1]+10, arcade.color.BLACK, font_size=15)


        # draw frontiers
        if self.debug_frontiers:
            pos = np.array(self.get_position()) + np.array(self.size_area)/2
            arcade.draw_text(str(self.identifier), pos[0], pos[1], map_id_to_color[self.identifier], 20)
            for frontierId, frontier in enumerate(self.frontiers):
                for point in frontier:
                    if frontierId == self.selected_frontier_id:
                        pos = np.array(self.map.grid_to_world(point)) + np.array(self.size_area)/2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 8, 8, map_id_to_color[self.identifier])
                    else:
                        pos = np.array(self.map.grid_to_world(point)) + np.array(self.size_area)/2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 2, 2, map_id_to_color[self.identifier])


    def draw_bottom_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_path:
            for k in range(len(self.path)-1):
                pt1 = self.path[k].array + np.array(self.size_area) / 2
                pt2 = self.path[k+1].array + np.array(self.size_area) / 2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(0, 255, 0))

            if self.next_waypoint is not None:
                pt1 = self.next_waypoint.array + np.array(self.size_area) / 2
                pt2 = self.drone_position.array + np.array(self.size_area) / 2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 0))

            if self.target is not None:
                pt = self.target.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pt[0], pt[1], 10, [255,0,255])

        if self.debug_kill_zones and self.kill_zone_mode:
            for killed_drone_pos in self.map.kill_zones.values():
                # draw a rectangle
                arcade.draw_rectangle_filled(killed_drone_pos.x + self.size_area[0]/2, killed_drone_pos.y + self.size_area[1]/2, 100, 100, arcade.color.RED)