"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
import numpy as np
from typing import Optional
from statemachine import StateMachine, State
import arcade
from collections import deque 

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean, normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor

from enum import Enum
import numpy as np
import cv2

from spg_overlay.utils.grid import Grid
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR


class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class Map:
    def __init__(self, area_world, resolution, lidar):
        
        self.resolution = resolution
        self.x_max_grid: int = int(area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(area_world[1] / self.resolution + 0.5)
        
        self.map = np.full((self.x_max_grid, self.y_max_grid), Zone.INEXPLORED)

        self.wall_grid = Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WALL)
        self.rescue_center_grid = Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.RESCUE_CENTER)
        self.wounded_grid = Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WOUNDED_PERSON)
        self.explored_grid = Mapper(area_world, resolution, lidar, None)

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone
        
    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]
    
    def update_grid(self, pose: Pose, semantic_lidar):
        self.wall_grid.update_grid(pose, semantic_lidar)
        self.rescue_center_grid.update_grid(pose, semantic_lidar)
        self.wounded_grid.update_grid(pose, semantic_lidar)
        self.explored_grid.update_grid(pose, semantic_lidar)

        # TODO: optimiser: mettre à jour uniquement les points qui ont changés
        # Update map
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                if self.wall_grid.binary_grid[x][y] == 1:
                    self[x,y] = Zone.OBSTACLE
                # elif self.rescue_center_grid.binary_grid[x][y] == 1:
                #     self.set_zone(x, y, Zone.RESCUE_CENTER)
                # elif self.wounded_grid.binary_grid[x][y] == 1:
                #     self.set_zone(x, y, Zone.WOUNDED)
                elif self.explored_grid.binary_grid[x][y] == 1:
                    self[x,y] = Zone.EMPTY
                else:
                    self[x,y] = Zone.INEXPLORED

    def display_map(self):
        """
        displays the map
        """
        map = self.map.copy()
        map = np.where(map == Zone.OBSTACLE, 255, map)
        map = np.where(map == Zone.WOUNDED, 128, map)
        map = np.where(map == Zone.RESCUE_CENTER, 64, map)
        map = np.where(map == Zone.INEXPLORED, 0, map)
        map = np.where(map == Zone.EMPTY, 128, map)
        map = map.astype(np.uint8)
        # save map as png
        cv2.imwrite("map.png", map.T)

class Mapper(Grid):

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 lidar,
                 zone_to_detect: DroneSemanticSensor.TypeEntity = DroneSemanticSensor.TypeEntity.WALL):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.lidar = lidar
        self.semantic_lidar = None

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.zone_to_detect = zone_to_detect

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.binary_grid = np.zeros((self.x_max_grid, self.y_max_grid)).astype(np.uint8)

    def update_grid(self, pose: Pose, semantic_lidar):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        EVERY_N = 1
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -16.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40


        self.lock_grid = self.grid == THRESHOLD_MIN
        self.buffer = self.grid.copy()
        self.confidence_grid = np.zeros((self.x_max_grid, self.y_max_grid))

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        if self.zone_to_detect:
            max_range = MAX_RANGE_LIDAR_SENSOR * 0.9
            # For empty zones
            # points_x and point_y contains the border of detected empty zone
            # We use a value a little bit less than LIDAR_DIST_CLIP because of the noise in lidar
            lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
            # All values of lidar_dist_empty_clip are now <= max_range
            lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
            points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
            points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

            for pt_x, pt_y in zip(points_x, points_y):
                self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

            # For obstacle zones, all values of lidar_dist are < max_range
            select_collision = lidar_dist < max_range

            points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
            points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)

            points_x = points_x[select_collision]
            points_y = points_y[select_collision]

            self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

            # the current position of the drone is free !
            self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)
        else:
            lidar_dist_empty_clip = np.minimum(lidar_dist, MAX_RANGE_LIDAR_SENSOR)
            points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
            points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

            for pt_x, pt_y in zip(points_x, points_y):
                self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, THRESHOLD_MAX)
            

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.grid[self.lock_grid] = self.buffer[self.lock_grid]

        # map to binary
        self.binary_grid = np.where(self.grid > 0, 1, 0)

class DroneController(StateMachine):

    # states  
    roaming = State('Roaming', initial=True)
    going_to_wounded = State('Going to wounded')
    approaching_wounded = State('Approaching wounded')
    going_to_center = State('Going to center')
    approaching_center = State('Approaching to center')

    force_transition = roaming.to(going_to_wounded, on='before_going_to_wounded')

    # transitions
    cycle = (
        roaming.to(approaching_wounded, cond="found_wounded", on="before_approaching_wounded") |
        going_to_wounded.to(approaching_wounded, cond="found_wounded", on="before_approaching_wounded") |

        going_to_wounded.to(roaming, cond="lost_route") |

        # if wounded captured by someone else
        approaching_wounded.to(roaming, cond="no_wounded") |

        approaching_wounded.to(going_to_center, cond="grasped_wounded", on="before_going_to_center") |

        going_to_center.to(approaching_center, cond="found_center", on='before_approaching_center') |

        approaching_center.to(roaming, cond="lost_wounded") |
        going_to_center.to(roaming, cond="lost_wounded",) |

        roaming.to(roaming) |
        going_to_wounded.to(going_to_wounded) |
        approaching_wounded.to(approaching_wounded) |
        going_to_center.to(going_to_center) |
        approaching_center.to(approaching_center)
    )
 

    def __init__(self, drone : DroneAbstract, debug_mode: bool = False):
        self.drone = drone
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.debug_mode = debug_mode

        super(DroneController, self).__init__()


    ## transitions conditions

    def found_wounded(self):
        return self.drone.found_wounded
    
    def lost_route(self):
        return self.drone.onRoute and self.drone.nextWaypoint is None
    
    def no_wounded(self):
        return not self.drone.found_wounded and not self.drone.base.grasper.grasped_entities
    
    def grasped_wounded(self):
        return len(self.drone.base.grasper.grasped_entities) > 0
    
    def found_center(self):
        return self.drone.found_center
    
    def lost_wounded(self):
        return not self.drone.base.grasper.grasped_entities
    
    
    ## actions

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        if not self.debug_mode: return
        message = ". " + message if message else ""
        print(f"Running {event} from {source.id} to {target.id}{message}")

    @roaming.enter
    def on_enter_roaming(self):
        # TODO: implement exploration
        self.drone.onRoute = False

    def before_going_to_wounded(self):
        self.drone.path = self.drone.get_path(self.drone.drone_position, 0)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)

    def before_approaching_wounded(self):
        self.drone.onRoute = False
    
    @approaching_wounded.enter
    def on_enter_approaching_wounded(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1

    def before_going_to_center(self):
        self.drone.path = self.drone.get_path(self.drone.drone_position, 1)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_center.enter
    def on_enter_going_to_center(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)
        self.command["grasper"] = 1

    def before_approaching_center(self):
        self.drone.onRoute = False
    
    @approaching_center.enter
    def enter_approaching_center(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1


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
        self.controller = DroneController(self, debug_mode=True)
        self.last_angles = deque() # queue of the last angles

        self.wounded_found = []
        self.wounded_distance = 60 # The distance between wounded person to be considered as the same

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = True
        
        self.controller.force_transition()
        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.map = Map(self.size_area, 4, self.lidar())


    def adapt_angle_direction(self, pos: list):
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.drone_angle != None:
            
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

        if np.linalg.norm(pos - self.nextWaypoint) < 20:
            return True
        return False
        

    # TODO: implement communication
    def define_message_for_all(self):
        pass


    # TODO: determine the position in no GPS zone
    def get_position(self):
        """
        returns the position of the drone
        """
        return self.measured_gps_position()
    

    # TODO: determine the angle with more precision
    def get_angle(self):
        """
        returns the angle of the drone
        """
        angle = self.measured_compass_angle()
        angle2 = self.measured_compass_angle()
        self.last_angles.append(angle)
        if len(self.last_angles) > 5:
            self.last_angles.popleft()
        return circular_mean(np.array(self.last_angles))
    

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

    # TODO: fix wounded search (sensor can miss wounded person at some frames) and make it more precise
    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        command = {"forward": 0.5,
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
        return [[250,150],[20, -200],[-200,150],[-260,138]]


    # TODO: implement beziers curves for turning, 45° forward movement, go directly towards waypoint and rotate during movement
    def get_control_from_path(self, pos):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angle_from_waypoint = self.adapt_angle_direction(pos)
        angle_from_waypoint = normalize_angle(angle_from_waypoint+math.pi/4)

        if angle_from_waypoint > 0.5:
            command["rotation"] =  1.0
        elif angle_from_waypoint < -0.5:
            command["rotation"] =  -1.0
        else:
            command["rotation"] = angle_from_waypoint

        command["forward"] = math.cos(angle_from_waypoint-math.pi/4)
        command["lateral"] = math.sin(angle_from_waypoint-math.pi/4)
        norm = max(abs(command["forward"]),abs(command["lateral"]))
        command["forward"] = command["forward"]/norm
        command["lateral"] = command["lateral"]/norm     
        
        if self.check_waypoint(pos):
            if len(self.path) > 0:
                self.lastWaypoint = self.nextWaypoint.copy()
                self.nextWaypoint = self.path.pop()
            else:
                self.nextWaypoint = None
                self.onRoute = False
        return command


    def control(self):

        self.found_wounded, self.found_center, self.command_semantic = self.process_semantic_sensor()
        self.drone_position = self.get_position()
        self.drone_angle = self.get_angle()

        self.controller.cycle()

        self.update_mapping()
            
        return self.controller.command
    
    def update_mapping(self):
        """
        updates the map
        """
        detection_semantic = self.semantic_values()
        self.estimated_pose = Pose(self.drone_position, self.drone_angle)
        #self.estimated_pose = Pose(np.asarray(self.true_position()), self.true_angle())
        max_vel_angle = 0.08

        # update map if angular velocity is not too high
        if abs(self.measured_angular_velocity()) < max_vel_angle:
            self.map.update_grid(self.estimated_pose, detection_semantic)
        self.map.display_map()


    def draw_top_layer(self):

        if self.debug_wounded:
            for wounded in self.wounded_found:
                pos = np.array(wounded["position"]) + np.array(self.size_area)/2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.GREEN_YELLOW)
                arcade.draw_circle_outline(pos[0], pos[1],self.wounded_distance, arcade.color.GREEN_YELLOW)


    def draw_bottom_layer(self):

        if self.debug_path: 
            drawn_path = self.path.copy()
            drawn_path.append(self.nextWaypoint)
            if self.lastWaypoint != None: drawn_path.append(self.lastWaypoint)

            for k in range(len(drawn_path)-1):
                pt1 = np.array(drawn_path[k]) + np.array(self.size_area)/2
                pt2 = np.array(drawn_path[k+1]) + np.array(self.size_area)/2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 255))