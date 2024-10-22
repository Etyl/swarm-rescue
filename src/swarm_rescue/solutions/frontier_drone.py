from __future__ import annotations

import math
import numpy as np
from typing import Optional, List, Tuple, Deque, Dict
import arcade
from collections import deque
from statemachine import exceptions

from spg_overlay.entities.drone_abstract import DroneAbstract  # type: ignore
from spg_overlay.utils.misc_data import MiscData  # type: ignore
from spg_overlay.utils.utils import circular_mean # type: ignore
from solutions.utils.utils import normalize_angle, map_id_to_color  # type: ignore
from spg_overlay.utils.pose import Pose # type: ignore
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor # type: ignore
from spg_overlay.utils.constants import RANGE_COMMUNICATION, MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR # type: ignore

from solutions.utils.types import DroneData, WoundedData, Vector2D # type: ignore
from solutions.localizer.localizer import Localizer # type: ignore
from solutions.mapper.mapper import Map # type: ignore
from solutions.drone_controller import DroneController # type: ignore
from solutions.roamer.roamer import RoamerController # type: ignore

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

        self.drone_prev_position : Optional[Vector2D] = None
        self.onRoute : bool = False # True if the drone is on the route to the waypoint
        self.path : List[Vector2D] = []
        self.waypoint_index : Optional[int] = None # The index to the next waypoint to go to
        self.drone_position : Vector2D = Vector2D(0, 0) # The position of the drone
        self.drone_angle : float = 0 # The angle of the drone
        self.drone_angle_offset : float = 0 # The angle offset of the drone that can be changed by the states
        self.found_wounded : bool = False # True if the drone has found a wounded person
        self.last_wounded_seen : int = 10000 # time (frames) since the wounded was last seen
        self.last_angles : Deque[float] = deque() # queue of the last angles
        self.last_positions : Deque[Vector2D] = deque() # queue of the last positions
        self.repulsion : Vector2D = Vector2D(0,0) # The repulsion vector from the other drones
        self.wall_repulsion : Vector2D = Vector2D(0,0) # The repulsion vector from the walls
        self.center_angle : Optional[float] = None
        self.is_near_center : bool = False # True if the drone is near the center and switches state
        self.on_center : bool = False # True if the drone is in the center and needs to stop to deliver wounded
        self.ignore_repulsion : int = 0 # timer to ignore the repulsion vector (>0 => ignore)

        self.rescue_center_position: Optional[Vector2D] = None
        self.wounded_found_list : List[WoundedData] = [] # the list of wounded persons found
        self.wounded_target : Optional[Vector2D] = None # The wounded position to go to
        self.drone_list : List[DroneData] = [] # The list of drones

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = True
        self.debug_positions = True
        self.debug_map = False
        self.debug_roamer = False
        self.debug_controller = False
        self.debug_mapper = False
        self.debug_lidar = False
        self.debug_repulsion = False
        self.debug_kill_zones = False
        self.debug_wall_repulsion = False
        self.debug_frontiers = False

        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.roaming = False
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}

        self.map : Map = Map(area_world=self.size_area, resolution=8, debug_mode=self.debug_mapper)
        self.roamer_controller : RoamerController = RoamerController(self, self.map, debug_mode=self.debug_roamer)

        self.localizer : Localizer = Localizer()
        self.theoretical_velocity : Vector2D = Vector2D(0, 0)

        self.controller : DroneController = DroneController(self, debug_mode=self.debug_controller)

        self.last_other_drones_position : Dict[int, Tuple[Vector2D, float]] = {}
        self.kill_zones : List[Vector2D] = []
        self.semantic_drone_pos : List[Vector2D] = []
        self.potential_kill_zones : List[Vector2D] = []
        self.kill_zone_mode = True

        self.point_of_interest = (0,0)
        self.frontiers : List[List[Vector2D]] = []
        self.selected_frontier_id : int = 0

        self.stuck_iteration : int = 0
        self.time : int = 0
        self.time_in_no_gps : int = 0


        # TEMP / DEBUG

        self.curr_proj: Optional[Vector2D] = None

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


    def adapt_angle_direction(self, pos: Vector2D) -> float:
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.next_waypoint is None:
            return 0

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

        waypoint_vect = (self.next_waypoint - pos).array
        ref_vect = np.array([1, 0])

        drone_angle = normalize_angle(self.get_angle())
        waypoint_angle = normalize_angle(angle(ref_vect, waypoint_vect))

        return normalize_angle(waypoint_angle - drone_angle)


    def check_waypoint(self, pos: Vector2D):
        """
        checks if the drone has reached the waypoint
        """
        next_waypoint: Vector2D = self.next_waypoint
        dist = pos.distance(next_waypoint)
        if len(self.path) == 0: return dist < 30

        v1 : Vector2D = next_waypoint - pos
        v2 : Vector2D = self.path[-1] - next_waypoint

        if v1.norm() == 0 or v2.norm() == 0:
            turning_angle = 0
        else:
            turning_angle = (v1@v2)/(v1.norm()*v2.norm())

        return dist < 15 + (1+turning_angle)*20


    def define_message_for_all(self):
        data = DroneData(
            id = self.identifier,
            position = self.get_position(),
            angle = self.get_angle(),
            vel_angle = self.compute_vel_angle(),
            wounded_target = self.wounded_target,
            map = self.map,
            semantic_values = self.semantic_values(),
            kill_zone_mode = self.kill_zone_mode,
            next_waypoint= self.next_waypoint,
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


    def optimise_localization(self):
        """
        optimises the localization of the drone using SLAM
        """
        # starting_pos = self.drone_position
        # starting_angle = self.drone_angle
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
                point = Vector2D(0,0)
                point.setX(starting_pos.x + posX + lidar_dist*math.cos(lidar_angle+starting_angle+angle))
                point.setY(starting_pos.y + posY + lidar_dist*math.sin(lidar_angle+starting_angle+angle))
                point = self.map.world_to_grid(point)
                if point.x < 0 or point.x >= self.map.width or point.y < 0 or point.y >= self.map.height:
                    continue
                #value -= self.map.occupancy_grid.get_grid()[int(point[0]),int(point[1])]
                value -= self.map.get_confidence_wall_map(int(point.x),int(point.y))
            return value

        mindx, mindy, mindangle = 0,0,0

        for k in range(30):
            dx, dy, dangle = np.random.normal(0,1), np.random.normal(0,1), np.random.normal(0,0.1)
            if Q([dx,dy,dangle]) < Q([mindx,mindy,mindangle]):
                mindx, mindy, mindangle = dx, dy, dangle

        self.drone_position = Vector2D(starting_pos.x + mindx, starting_pos.y + mindy)
        self.drone_angle = starting_angle + mindangle


    # TODO: improve angle estimation
    def update_localization(self) -> None:
        """
        returns the position of the drone
        """

        rot = self.command["rotation"]
        measured_angle = self.measured_compass_angle()
        if measured_angle is not None:
            self.drone_angle = measured_angle
            self.time_in_no_gps = 0
        else:
            self.drone_angle = self.drone_angle + 0.2*rot
            #self.drone_angle += self.odometer_values()[2]

        measured_position = self.measured_gps_position()

        angle = self.drone_angle
        command = Vector2D(self.command["forward"], self.command["lateral"])
        command.rotate(angle)

        theoretical_velocity = self.theoretical_velocity + ((command * 0.56) - (self.theoretical_velocity * 0.095))
        v = self.odometer_values()[0]

        if measured_position is not None and abs(v) > 5:
            self.theoretical_velocity = Vector2D((v * math.cos(angle) + theoretical_velocity.x) / 2, (v * math.sin(angle) + theoretical_velocity.y) / 2)
            theoretical_position = self.drone_position + self.theoretical_velocity
            self.drone_position = (self.measured_gps_position() + theoretical_position) / 2
        elif measured_position is not None:
            self.theoretical_velocity = Vector2D(pointList=np.array([v * math.cos(angle), v * math.sin(angle)]) / 2)
            self.drone_position =  self.measured_gps_position()
        else:
            self.drone_position.setX(self.drone_position.x + self.odometer_values()[0] * np.cos(self.drone_angle + self.odometer_values()[1]))
            self.drone_position.setY(self.drone_position.y + self.odometer_values()[0] * np.sin(self.drone_angle + self.odometer_values()[1]))
            # self.theoretical_velocity = theoretical_velocity
            # self.drone_position = self.drone_position + self.theoretical_velocity
            self.optimise_localization()

    def measured_gps_position(self) -> Optional[Vector2D]:
        measured_position = super().measured_gps_position()
        if measured_position is None:
            return None
        return Vector2D(pointList=measured_position)

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

        """
        # update the visibility of the drones
        for data in self.drone_list:
            data.visible = False
        #drone_data.visible = True
        """

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


    def get_control_from_semantic(self):

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

        angular_vel_controller_max = 1.0

        command = {"forward": 1.0,
                    "lateral": 0.0,
                    "rotation": 0.0,
                    "grasper": 0}

        if self.found_wounded:
            best_angle = normalize_angle(angle(np.array([1,0]), self.wounded_target.array - self.drone_position.array))
            best_angle = normalize_angle(best_angle - normalize_angle(self.get_angle()))

            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.4

        if self.found_center and self.center_angle is not None:
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

        if self.found_center and self.on_center:
            command["forward"] = 0
            command["rotation"] = 0.5

        return command

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        detection_semantic = self.semantic_values()

        self.clear_wounded_found()

        self.last_wounded_seen += 1
        if detection_semantic:
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    # wounded: WoundedData = WoundedData(data)
                    self.add_wounded(data)

        found_rescue_center = False
        is_near,on_center = False,False
        angles_list = []
        if (detection_semantic and
            (self.controller.current_state == self.controller.going_to_center
            or self.controller.current_state == self.controller.approaching_center)):
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance <= MAX_RANGE_SEMANTIC_SENSOR*(3/4))
                    on_center = (data.distance <= 50)

        if found_rescue_center:
            self.center_angle = circular_mean(np.array(angles_list))
            self.is_near_center = is_near
            self.on_center = on_center
        else:
            self.center_angle = None
            self.is_near_center = False
            self.on_center = False


    def get_path(self, pos:Vector2D):
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

        if self.next_waypoint is not None and self.check_waypoint(pos):
            if self.check_waypoint(pos):
                if self.waypoint_index < len(self.path)-1:
                    self.waypoint_index += 1
                else:
                    self.path = []
                    self.waypoint_index = None
                    self.onRoute = False
        self.update_waypoint_index()

        return command


    def update_waypoint_index(self) -> None:
        if self.waypoint_index is None:
            return

        curr_proj: Optional[Vector2D] = None
        if 0 < self.waypoint_index <= len(self.path) - 1 and self.path[self.waypoint_index-1] != self.path[self.waypoint_index]:
            curr_proj = self.get_position().project(self.path[self.waypoint_index-1],self.path[self.waypoint_index])
            curr_proj += (self.path[self.waypoint_index]-self.path[self.waypoint_index-1]).normalize()*80

        self.curr_proj = curr_proj



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
                repulsion += Vector2D(math.cos(angle), math.sin(angle)) * repulsion_dist(dist)

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

        self.path = []
        self.waypoint_index = None
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
            self.last_other_drones_position[drone.id] = (drone.position, drone.vel_angle)

        # check if other drones are killed by checking it's not in drone_list anymore and have last seen distance < MAX_RANGE_LIDAR_SENSOR / 2
        killed_ids = []
        for droneId in self.last_other_drones_position:
            if droneId not in [drone.id for drone in self.drone_list]:
                if (self.last_other_drones_position[droneId][0].distance(self.drone_position)) < RANGE_COMMUNICATION * 0.85:
                    #print(f"Drone {droneId} killed")
                    self.path = []
                    self.waypoint_index = None
                    kill_zone_x = self.last_other_drones_position[droneId][0].x + 50*math.cos(self.last_other_drones_position[droneId][1])
                    kill_zone_y = self.last_other_drones_position[droneId][0].y + 50*math.sin(self.last_other_drones_position[droneId][1])
                    self.map.add_kill_zone(droneId, Vector2D(kill_zone_x, kill_zone_y))
                    killed_ids.append(droneId)
                else:
                    #print(f"Drone {droneId} left")
                    killed_ids.append(droneId)

        for droneId in killed_ids:
            self.last_other_drones_position.pop(droneId)

        for droneId, kill_zone in self.map.kill_zones.items():
            if droneId in [drone.id for drone in self.drone_list]:
                self.kill_zone_mode = False


    def control(self):

        # check if drone is dead
        if self.odometer_values() is None:
            return

        self.stuck_iteration += 1
        self.time += 1
        self.update_localization()
        if self.gps_disabled:
            self.time_in_no_gps += 1
        self.process_semantic_sensor()
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

        self.drone_prev_position = self.drone_position.copy()

        self.point_of_interest = self.compute_point_of_interest()

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


    def set_selected_frontier_id(self, frontierId : int):
        """
        sets the selected frontier id
        """
        self.selected_frontier_id = frontierId

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

            direction = self.theoretical_velocity
            arcade.draw_line(pos[0], pos[1], pos[0]+direction.x*20, pos[1]+direction.y*20, arcade.color.GREEN)

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

            if self.curr_proj is not None:
                pt = self.curr_proj.array + np.array(self.size_area) / 2
                arcade.draw_circle_filled(pt[0], pt[1], 10, [255,0,255])



        if self.debug_kill_zones and self.kill_zone_mode:
            for killed_drone_pos in self.map.kill_zones.values():
                pos = np.array(killed_drone_pos) + np.array(self.size_area)/2
                # draw a rectangle
                arcade.draw_rectangle_filled(pos[0], pos[1], 100, 100, arcade.color.RED)
