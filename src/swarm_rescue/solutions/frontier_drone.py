from __future__ import annotations

import math
import os.path

import numpy as np
from typing import Optional, List, Tuple, Deque, Dict
import arcade
from collections import deque, defaultdict

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
from solutions.mapper.mapper import Map, DRONE_SIZE_RADIUS
from solutions.mapper.graph_map_class import GraphMap, MAX_CELL_SIZE, MAX_CELL_RADIUS
from solutions.drone_controller import DroneController
from solutions.roamer.roamer import RoamerController


class FrontierDrone(DroneAbstract):

    POSITION_QUEUE_SIZE = 30 # number of positions to check if the drone is stuck
    REFRESH_PATH_LIMIT = 40 # frames before refreshing the path
    WOUNDED_DISTANCE = 50 # The distance between wounded person to be considered as the same
    START_IDLE_TIME = 40 # Time before the drone starts moving
    SAME_DRONE_THRESHOLD = 100 # The distance between two drones to be considered as the same

    def __init__(
        self,
        identifier: Optional[int] = None,
        misc_data: Optional[MiscData] = None,
        save_dir = None,
        **kwargs):

        super().__init__(
            identifier=identifier,
            misc_data=misc_data,
            display_lidar_graph=False,
            **kwargs)

        self.START_IDLE_TIME = FrontierDrone.START_IDLE_TIME

        self.path : List[Vector2D] = []
        self.drone_angle_offset : float = 0 # The angle offset of the drone that can be changed by the states
        self.found_wounded : bool = False # True if the drone has found a wounded person
        self.last_positions : Deque[Vector2D] = deque()
        self.last_wounded_seen : int = 10000 # time (frames) since the wounded was last seen
        self.repulsion_drone : Vector2D = Vector2D() # The repulsion vector from the other drones
        self.drone_direction_group = Vector2D() # Normalised repulsion vector to group of drones
        self.repulsion_wall : Vector2D = Vector2D() # The repulsion vector from the walls
        self.closest_wall : Vector2D = Vector2D()
        self.center_angle : Optional[float] = None # angle from the visible rescue center
        self.rescue_center_dist : Optional[float] = None # distance from the visible rescue center
        self.ignore_repulsion : int = 0 # timer to ignore the repulsion vector (>0 => ignore)

        self.command_pos: Vector2D = Vector2D()
        self.visible_drones: List[Tuple[int,Vector2D]] = [] # List of (id,drone_position)
        self.searching_time : int = 0 # time spent searching for next target
        self.return_zone_position : Optional[Vector2D] = None # The position of the return zone
        self.no_comm_zone_mode : bool = False # True if the drone is in the no communication zone

        self.rescue_center_position: Optional[Vector2D] = None
        self.wounded_found_list : List[WoundedData] = [] # the list of wounded persons found
        self.wounded_target : Optional[Vector2D] = None # The wounded position to go to
        self.drone_list : List[DroneData] = [] # The list of drones
        self.drone_positions: Dict[int,Tuple[int,Vector2D]] = dict() # drone_id : (timestep, drone_position)
        self.iteration : int = 0 # The number of iterations
        self.old_exploration_score : float = 0 # The old exploration score
        self.old_drone_exploration_score : float = 0 # The old drone exploration score

        ## Debug controls

        self.debug_path = False # True if the path must be displayed
        self.debug_wounded = False
        self.debug_positions = False
        self.debug_roamer = False
        self.debug_controller = False
        self.debug_mapper = False
        self.debug_lidar = False
        self.debug_repulsion = False
        self.debug_wall_repulsion = False
        self.debug_frontiers = False
        self.debug_graph_map = False

        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self._command: Dict[str, float] = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }
        self.prev_command: Dict[str, float] = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }

        self.map : Map = Map(area_world=self.size_area, resolution=4, identifier=self.identifier, debug_mode=self.debug_mapper)

        if save_dir:
            save_file = os.path.join(save_dir, f"data_{self.identifier}.txt")
        else:
            save_file = None
        current_dir = os.path.dirname(__file__)
        self.graph_map = GraphMap(map=self.map, resolution=2, filename=save_file,gqn_file=os.path.join(current_dir,"data/best_gqn_0188.pth"))
        self.roamer_controller : RoamerController = RoamerController(self, self.map, debug_mode=self.debug_roamer)

        self.localizer : Localizer = Localizer(self)

        self.controller : DroneController = DroneController(self, debug_mode=self.debug_controller)

        self.point_of_interest = (0,0)
        self.frontiers : List[List[Vector2D]] = []
        self.selected_frontier_id : int = 0
        self.last_frontier_target : Optional[Vector2D] = None

        self.stuck_iteration : int = 0
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
        return self.last_wounded_seen < 5

    @property
    def next_waypoint(self) -> Optional[Vector2D]:
        return self.localizer.next_waypoint

    @property
    def target(self) -> Optional[Vector2D]:
        return self.localizer.target

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


    def define_message_for_all(self):
        if len(self.localizer.path)>0:
            target_path = self.localizer.path[-1]
        else:
            target_path = None
        data = DroneData(
            id = self.identifier,
            position = self.drone_position,
            angle = self.drone_angle,
            vel_angle = self.localizer.drone_velocity_angle,
            wounded_target = self.wounded_target,
            map = self.map,
            semantic_values = self.semantic_values(),
            next_waypoint= self.next_waypoint,
            confidence_position = self.localizer.confidence_position,
            visible_drones = self.visible_drones,
            rescue_center_position=self.rescue_center_position,
            no_comm_zone_mode=self.no_comm_zone_mode,
            target=target_path,
            drone_positions=self.drone_positions
        )
        return data

    def add_wounded(self, data_wounded):
        """
        compares the wounded persons detected with the ones already detected
        """

        def get_wounded_position():
            pos = self.drone_position.copy()
            angle = normalize_angle(self.drone_angle + data_wounded.angle)
            pos.setX(pos.x + data_wounded.distance * math.cos(angle))
            pos.setY(pos.y + data_wounded.distance * math.sin(angle))
            return pos

        # check if wounded target is visible
        wounded_pos = get_wounded_position()


        if self.wounded_target is not None and wounded_pos.distance(self.wounded_target) < FrontierDrone.WOUNDED_DISTANCE:
            if self.drone_position.distance(self.wounded_target)<2*DRONE_SIZE_RADIUS or self.map.is_reachable(self.drone_position, self.wounded_target):
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
            if self.drone_position.distance(self.wounded_found_list[k].position) < FrontierDrone.WOUNDED_DISTANCE and self.wounded_found_list[k].last_seen > 5:
                self.wounded_found_list.pop(k)

    def update_drones(self, drone_data : DroneData):
        """
        updates the data of the drones
        """
        # update drone positions
        self.drone_positions[drone_data.id] = (self.elapsed_timestep,drone_data.position)
        for drone_id in drone_data.drone_positions:
            if drone_id not in self.drone_positions:
                self.drone_positions[drone_id] = drone_data.drone_positions[drone_id]
            elif drone_data.drone_positions[drone_id][0] > self.drone_positions[drone_id][0]:
                self.drone_positions[drone_id] = drone_data.drone_positions[drone_id]

        if self.rescue_center_position is None and drone_data.rescue_center_position is not None:
            self.rescue_center_position = drone_data.rescue_center_position
        
        # set the no communication zone mode to True if one of the drones is in this mode
        if drone_data.no_comm_zone_mode:
            self.map.reset_kill_zone()
            self.no_comm_zone_mode = True

        # update the wounded list
        if drone_data.wounded_target is not None:
            for k in range(len(self.wounded_found_list)):
                if self.wounded_found_list[k].position.distance(drone_data.wounded_target) < FrontierDrone.WOUNDED_DISTANCE:
                    #self.wounded_found_list[k].taken = True
                    self.wounded_found_list[k].drone_taker = drone_data.id
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
                distance = self.drone_position.distance(wounded.position)

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


    def get_path(self, pos:Vector2D) -> Optional[List[Vector2D]]:
        """
        returns the path to the destination
        """
        return self.map.shortest_path(self.drone_position, pos)[0]


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

        def repulsion_coef(dist:float) -> float:
            return 2*max(0,min(1,((MAX_RANGE_SEMANTIC_SENSOR-dist+20)/MAX_RANGE_SEMANTIC_SENSOR)))**2

        found_pos = []
        def add_pos(p:Vector2D) -> bool:
            for p2 in found_pos:
                if p.distance(p2) < 30:
                    return False
            found_pos.append(p)
            return True

        repulsion = Vector2D()
        min_dist = np.inf
        min_angle = 0
        for data in self.semantic_values():
            if (
                data.entity_type == DroneSemanticSensor.TypeEntity.DRONE or (
                data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and
                data.grasped == True and
                self.controller.current_state not in [self.controller.going_to_center, self.controller.approaching_center]
            )):
                angle = data.angle
                dist = data.distance
                if dist < min_dist:
                    min_dist = dist
                    min_angle = angle

                pos = self.drone_position + dist * Vector2D(1,0).rotate(angle)
                if not add_pos(pos):
                    continue

                repulsion += -repulsion_coef(dist) * Vector2D(math.cos(angle), math.sin(angle))

        centroid = self.drone_position
        for p in found_pos:
            centroid += p
        centroid = centroid/(len(found_pos)+1)
        self.drone_direction_group = (self.drone_position-centroid).normalize()

        if repulsion.norm() == 0:
            self.repulsion_drone = Vector2D(0, 0)
            return

        repulsion = repulsion.normalize()
        repulsion *= min(2.0,repulsion_coef(min_dist-20)+0.2)

        if self.controller.current_state in [self.controller.going_to_center, self.controller.approaching_wounded, self.controller.approaching_center]:
            self.repulsion_drone = 0.05 * repulsion
        elif self.controller.current_state == self.controller.going_to_return_zone:
            if self.drone_position.distance(self.return_zone_position) <= 400:
                self.repulsion_drone = 0.05 * repulsion
            else:
                self.repulsion_drone = 0.3 * repulsion
        elif self.controller.current_state == self.controller.stay_in_return_zone:
            self.repulsion_drone = 0 * repulsion
        else:
            if min_dist < 70:
                repulsion_norm: Vector2D = -Vector2D(1,0).rotate(min_angle)
                self.command_pos += -(self.command_pos @ repulsion_norm) * repulsion_norm
                if repulsion.norm()>1: repulsion = repulsion.normalize()
            self.repulsion_drone = repulsion

        if self.ignore_repulsion <= 0:
            self.command_pos += self.repulsion_drone
        else:
            self.ignore_repulsion -= 1


    def update_wall_repulsion(self) -> None:
        """
        update repulsion vector for the drone from the walls (local drone vector)
        """

        if self.controller.current_state == self.controller.stay_in_return_zone:
            return

        lidar_dist = self.lidar().get_sensor_values()
        lidar_angles = self.lidar().ray_angles
        detection_semantic = self.semantic_values()

        angles_available_arr = [True]*181 # angles from -pi to +pi
        for data in detection_semantic:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER and
                self.controller.current_state != self.controller.approaching_center):
                continue

            i = int(round(180*((np.pi+data.angle)/(2*np.pi))))
            for k in range((i-3)%181,(i+4)%181):
                angles_available_arr[k]=False

        repulsion_vectors : List[Vector2D] = [Vector2D(0,0)]

        for i in range(len(lidar_dist)):
            if (angles_available_arr[i] and
                lidar_dist[i] < 0.3 * MAX_RANGE_LIDAR_SENSOR):
                    d = 1 - lidar_dist[i]/MAX_RANGE_LIDAR_SENSOR
                    repulsion_vectors.append(Vector2D(d*math.cos(lidar_angles[i]), d*math.sin(lidar_angles[i])))

        # check if drone is too close to a wall
        kmin = np.argmax([v.norm() for v in repulsion_vectors])
        self.repulsion_wall = Vector2D()
        self.closest_wall = repulsion_vectors[kmin]

        if repulsion_vectors[kmin].norm() >= 0.91:
            rep:Vector2D = repulsion_vectors[kmin].normalize()
            self.repulsion_wall = -(self.command_pos @ rep) * rep
            self.repulsion_wall += -repulsion_vectors[kmin].normalize()
        elif repulsion_vectors[kmin].norm() >= 0.8:
            v =  self.localizer.drone_velocity.rotate(-self.localizer.drone_angle) @ repulsion_vectors[kmin].normalize()
            c = -max(0,1.3*np.tanh(v/3))
            self.repulsion_wall = repulsion_vectors[kmin].normalize() * c

        self.command_pos += self.repulsion_wall
        # TODO : change repulsion according to drone direction (change forward and sideways command)


    def update_repulsion(self) -> None:
        self.command_pos = Vector2D(self._command["forward"], self._command["lateral"])

        self.update_drone_repulsion()
        self.update_wall_repulsion()

        if self.command_pos.norm()>1:
            self.command_pos = self.command_pos.normalize()
        self._command["forward"] = self.command_pos.x
        self._command["lateral"] = self.command_pos.y

    # TODO: rewrite
    def test_stuck(self):

        if self.target is not None and self.controller.current_state in [self.controller.going_to_center, self.controller.going_to_wounded, self.controller.roaming]:
            if not self.map.is_reachable(self.drone_position, self.target):
                self.reset_path()

        self.last_positions.append(self.drone_position)
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
            if closest_drone is None or drone.position.distance(self.drone_position) < dist:
                dist = drone.position.distance(self.drone_position)
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
        self.localizer.reset_path()

    def set_path(self, path: Optional[List[Vector2D]]) -> None:
        """
        set a new path
        """
        self.localizer.set_path(path)

    def compute_kill_zone(self):
        """
        computes the kill zone
        """
        detection_semantic = self.semantic_values()

        for data in detection_semantic:
            if (data.entity_type == DroneSemanticSensor.TypeEntity.DRONE):
                angle = data.angle
                dist = data.distance

                pos = self.drone_position + dist * Vector2D(math.cos(angle + self.drone_angle), math.sin(angle + self.drone_angle))
                # Check if this drone is communicating
                dead = True
                for drone in self.drone_list:
                    if drone.id == self.identifier: continue
                    if drone.position.distance(pos) < self.SAME_DRONE_THRESHOLD:
                        dead = False
                        break
                if dead:
                    self.map.add_kill_zone(pos)

    def add_searching_time(self):
        """
        adds the searching time
        """
        self.searching_time += 1
    
    def reset_searching_time(self):
        """
        resets the searching time
        """
        self.searching_time = 0
    
    def is_simulation_time_limit_reached(self):
        """
        checks if the simulation time limit is reached
        """
        max_timestep = self._misc_data.max_timestep_limit * 0.85
        max_wall_time = self._misc_data.max_walltime_limit * 0.85
        return self.elapsed_timestep >= max_timestep or self.elapsed_walltime >= max_wall_time
    
    def update_reward(self):
        """
        updates the reward with current exploration - old reward
        """
        self.old_drone_exploration_score = self.map.drone_exploration_score
        self.old_exploration_score = self.map.exploration_score
    
    def get_reward(self):
        """
        returns the current reward
        """
        return self.map.exploration_score, self.map.drone_exploration_score, self.old_exploration_score, self.old_drone_exploration_score

    def get_path_info(self) -> List[float]:
        info = [
            self.elapsed_timestep,
            self.drone_position.distance(self.last_frontier_target) if self.last_frontier_target else -1,
            self.graph_map.map_width,
            self.graph_map.map_height,
            MAX_CELL_SIZE,
            MAX_CELL_RADIUS
        ]
        return info

    def control(self):
        # check if drone is dead
        if self.odometer_values() is None:
            return
        
        if self.communicator_is_disabled():
            self.map.reset_kill_zone()
            self.no_comm_zone_mode = True
        
        self.stuck_iteration += 1
        if self.gps_disabled:
            self.time_in_no_gps += 1


        self.localizer.localize()
        self.process_semantic_sensor()
        self.process_communicator()
        self.check_wounded_available()
        self.test_stuck()

        if self.return_zone_position is None:
            self.return_zone_position = self.drone_position.copy()

        if self.rescue_center_position is None:
            self.compute_rescue_center_position()

        if not self.communicator_is_disabled() and not self.no_comm_zone_mode:
            if self.iteration > 10:
                self.compute_kill_zone()
        self.update_last_other_drones_position()

        if self.roaming:
            try:
                self.roamer_controller.cycle()
            except exceptions.TransitionNotAllowed:
                pass

        self.controller.cycle()

        self.roaming = self.controller.current_state == self.controller.roaming

        self.update_mapping()

        if self.elapsed_timestep < FrontierDrone.START_IDLE_TIME:
            self._command = {
                "rotation":0,
                "forward":0,
                "lateral":0
            }
        elif self.roaming:
            self._command = self.roamer_controller.command.copy()
        else:
            self._command = self.controller.command.copy()

        self.update_repulsion()

        if self.gps_disabled and self.controller.current_state != self.controller.going_to_center:
            self._command["rotation"] /=2
            self._command["forward"] /=2
            self._command["lateral"] /=2
        elif self.controller.current_state == self.controller.stay_in_return_zone:
            self._command["rotation"] /= 4
            self._command["forward"] /= 4
            self._command["lateral"] /= 4


        self.point_of_interest = self.compute_point_of_interest()
        self.prev_command = self._command
        self.previous_drone_health = self.drone_health

        self.iteration += 1

        return self._command


    # TODO : update confidence map using the velocity
    def update_mapping(self) -> None:
        """
        updates the map
        """
        estimated_pose = Pose(self.drone_position.array, self.drone_angle)

        if not self.gps_disabled:
            self.map.update(estimated_pose, self.lidar(), self)
        else:
            max_vel_angle = 0.08
            if abs(self.measured_angular_velocity()) < max_vel_angle:
                self.map.update_confidence_wall_map()
                self.map.update(estimated_pose, self.lidar(), self)

        merging_maps:List[Map] = []
        for other_drone in self.drone_list:
            if other_drone.id == self.identifier: continue
            merging_maps.append(other_drone.map)

        self.map.merge(merging_maps, self)


    def set_selected_frontier_id(self, frontier_id : int):
        """
        sets the selected frontier id
        """
        self.selected_frontier_id = frontier_id

    def draw_top_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return

        if self.debug_graph_map:
            if self.identifier==4:
                self.graph_map.draw(self)

            if self.last_frontier_target is not None:
                pt = self.last_frontier_target.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pt[0], pt[1], 10, [255, 0, 255])

            # draw reward
            pos = self.drone_position.array + np.array(self.size_area) / 2
            arcade.draw_text(f"Reward: {round(self.get_reward()[1],2)}", pos[0], pos[1], arcade.color.BLACK, font_size=15)

        # draw frontiers
        if self.debug_frontiers:
            pos = self.drone_position.array + np.array(self.size_area) / 2
            for frontierId, frontier in enumerate(self.frontiers):
                for point in frontier:
                    if frontierId == self.selected_frontier_id:
                        pos = self.map.grid_to_world(point).array + np.array(self.size_area) / 2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 8, 8, map_id_to_color[self.identifier])
                    else:
                        pos = self.map.grid_to_world(point).array + np.array(self.size_area) / 2
                        arcade.draw_rectangle_filled(pos[0], pos[1], 2, 2, map_id_to_color[self.identifier])

        if self.debug_repulsion:
            pos = self.drone_position.array + np.array(self.size_area) / 2
            repulsion = 2*self.repulsion_drone.rotate(self.drone_angle)
            arcade.draw_line(pos[0], pos[1], pos[0]+repulsion.x*20, pos[1]+repulsion.y*20, arcade.color.RED)

            repulsion = 2*self.repulsion_wall.rotate(self.drone_angle)
            arcade.draw_line(pos[0], pos[1], pos[0] + repulsion.x * 20, pos[1] + repulsion.y * 20, arcade.color.GREEN)

            repulsion = 2*self.closest_wall.rotate(self.drone_angle)
            arcade.draw_line(pos[0], pos[1], pos[0] + repulsion.x * 20, pos[1] + repulsion.y * 20, arcade.color.BLUE)

        if self.debug_wall_repulsion:
            pos = self.drone_position.array + np.array(self.size_area) / 2
            wall_repulsion = self.repulsion_wall.rotate(self.drone_angle)
            arcade.draw_line(pos[0], pos[1], pos[0]+wall_repulsion.x*0.25*MAX_RANGE_LIDAR_SENSOR, pos[1]+wall_repulsion.y*0.25*MAX_RANGE_LIDAR_SENSOR, arcade.color.RED)

        if self.debug_lidar:
            lidar_dist = self.lidar().get_sensor_values()[::].copy()
            lidar_angles = self.lidar().ray_angles[::].copy()
            for k in range(len(lidar_dist)):
                pos = self.drone_position.array + np.array(self.size_area) / 2
                pos[0] += lidar_dist[k]*math.cos(lidar_angles[k]+self.drone_angle)
                pos[1] += lidar_dist[k]*math.sin(lidar_angles[k]+self.drone_angle)
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
            pos = self.drone_position.array + np.array(self.size_area) / 2
            if self._command["grasper"] == 1:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.RED)
            else:
                arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.GREEN)

            direction = np.array([1,0])
            rot = np.array([[math.cos(self.drone_angle), math.sin(self.drone_angle)],[-math.sin(self.drone_angle), math.cos(self.drone_angle)]])
            direction = direction@rot
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*50, pos[1]+direction[1]*50, arcade.color.RED)
            arcade.draw_text(f"{self.localizer.confidence_position:.3f}", pos[0]+10, pos[1]+10, arcade.color.BLACK, font_size=15)

        debug = False
        if debug:
            # draw return zone
            if self.return_zone_position is not None:
                pos = self.return_zone_position.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pos[0], pos[1], 10, arcade.color.BLUE)

            # draw drone health over the drone
            pos = self.drone_position.array + np.array(self.size_area) / 2
            arcade.draw_text(f"{self.drone_health}", pos[0], pos[1], arcade.color.BLACK, font_size=15)

        if self.debug_frontiers:
            pos = self.drone_position.array + np.array(self.size_area) / 2
            arcade.draw_text(str(self.identifier), pos[0], pos[1], map_id_to_color[self.identifier], 20)

    def draw_bottom_layer(self):
        # check if drone is dead
        if self.odometer_values() is None: return



        if self.debug_path:
            for k in range(len(self.localizer.path)-1):
                pt1 = self.localizer.path[k].array + np.array(self.size_area) / 2
                pt2 = self.localizer.path[k+1].array + np.array(self.size_area) / 2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(0, 255, 0))

            if self.next_waypoint is not None:
                pt1 = self.next_waypoint.array + np.array(self.size_area) / 2
                pt2 = self.drone_position.array + np.array(self.size_area) / 2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 0))

            if self.localizer.target is not None:
                pt = self.localizer.target.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pt[0], pt[1], 10, [255,0,255])
