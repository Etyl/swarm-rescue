"""
Roamer module
implements roaming behavior of the drone
"""
from __future__ import annotations

from solutions.mapper.mapper import Map  # type: ignore
from solutions.mapper.mapper import Zone  # type: ignore
from solutions.roamer.frontier_explorer import get_frontiers  # type: ignore
from solutions.utils.constants import FRONTIER_COUNT # type: ignore
from solutions.utils.types import Vector2D # type: ignore
from solutions.utils.utils import normalize_angle # type: ignore

from statemachine import StateMachine, State
from typing import TYPE_CHECKING, Optional, Tuple, List, Dict
import numpy as np
import os
import cv2
import matplotlib.pyplot as plt # type: ignore

if TYPE_CHECKING: # type: ignore
    from solutions.frontier_drone import FrontierDrone # type: ignore

# OTHER IMPL - ASYNC
# import asyncio
# import threading


MAP_SCALE : int = 2

class RoamerController(StateMachine):

    # maximum number of times the drone can be in the same position
    # i.e. the maximum number of times the check_target_reached function can return False
    _LOOP_COUNT_GOING_TO_TARGET_THRESHOLD : int = 50

    # maximum number of times the drone can be close to the previous searching start point
    # i.e. the maximum number of times the test_position_close_start_point function can return True
    _COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD : int= 50

    # maximum number of times the drone can't find a target
    # i.e. the maximum number of times the search_for_target function can return None
    _NONE_TARGET_FOUND_THRESHOLD : int = 10

    # the thickness of the walls in the map when the path is computed (in order to allow a smoother path)
    _WALL_THICKENING : int = 3

    # the sampling rate of the path (in order to reduce the number of points)
    _PATH_SAMPLING_RATE : int = 1

    # the number of points required in a frontier
    _FRONTIERS_THRESHOLD : int = 2

    # the minimum number of points required in a frontier
    _MIN_FRONTIERS_THRESHOLD : int = 2

    # the cooldown for computing the frontier
    # TODO rewrite/refactor
    _COMPUTE_FRONTIER_COOLDOWN : int = 10

    start_roaming = State('Start Roaming', initial=True)
    searching_for_target = State('Searching for target')
    going_to_target = State('Going to target')

    force_searching_for_target = start_roaming.to(searching_for_target, on='before_searching_for_target')
    force_going_to_target = searching_for_target.to(going_to_target, on='before_going_to_target')

    cycle = (
        start_roaming.to(searching_for_target, cond="drone_position_valid") |

        searching_for_target.to(going_to_target, cond="target_discovered") |
        searching_for_target.to(searching_for_target) |

        going_to_target.to(searching_for_target, cond="no_path_to_target") |
        going_to_target.to(searching_for_target, cond="check_target_reached") |
        going_to_target.to(going_to_target)
    )

    def __init__(self, drone: FrontierDrone, drone_map: Map, debug_mode: bool = False) -> None:
        self.drone : FrontierDrone = drone
        self.map : Map = drone_map
        self.command = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }
        self.debug_mode: bool = debug_mode
        self.roamer: Roamer = Roamer(drone, drone_map, debug_mode)

        self.loop_count_going_to_target: int = 0

        self.target : Optional[Vector2D] = None
        self.none_target_count : int = 0
        
        # numbers of points required in a frontier
        # the idea is to decrease this number if the drone is stuck at some point
        # because it would mean that the drone has already explored a lot of the map
        # and that the remaining unexplored areas are small
        self.frontiers_threshold : int = self._FRONTIERS_THRESHOLD

        self.previous_searching_start_point : Optional[Vector2D] = None
        self.count_close_previous_searching_start_point : int= 0

        self.waiting_time : int = 0
        self.last_time_updates : int = self._COMPUTE_FRONTIER_COOLDOWN
        self.first_time : bool = True
        self.first_time_cooldown: int = 0

        super(RoamerController, self).__init__()

    def no_path_to_target(self):
        # if the counter reaches a certain threshold, we restart the search
        if self.loop_count_going_to_target >= self._LOOP_COUNT_GOING_TO_TARGET_THRESHOLD: return True

        if self.count_close_previous_searching_start_point >= self._COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD: return True

        # in case the drone has ended its path
        if self.drone.next_waypoint is None: return True

        # if there is no target
        if self.target is None: return True

        return False
    
    def check_target_reached(self) -> bool:
        """
        checks if the drone has reached the waypoint
        """
        if len(self.drone.localizer.path) == 0:
            return False

        dist = self.drone.drone_position.distance(self.target)
        if len(self.drone.path) == 0: return dist < 20

        v1 : Vector2D = self.target- self.drone.drone_position

        v2 : Vector2D = self.drone.path[-1] - self.map.grid_to_world(self.target)
        
        if v1.norm() == 0 or v2.norm() == 0:
            turning_angle = 0
        else: 
            turning_angle = (v1@v2)/(v1.norm()*v2.norm())

        return dist < 20 + (1+turning_angle)*20

    def drone_position_valid(self):
        return self.drone.elapsed_timestep > self.drone.START_IDLE_TIME

    def target_discovered(self):
        if self.debug_mode: print(f"Target discovered: {self.target}")
        return self.target is not None

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        if self.debug_mode: print(f"Running {event} from {source.id} to {target.id}{message}")
        return f"Running {event} from {source.id} to {target.id}{message}"
    
    def before_going_to_target(self):
        self.loop_count_going_to_target = 0 # reset position counter

        self.none_target_count = 0 # reset none target counter


    def before_searching_for_target(self):
        pass

    def search_for_target(self):
        self.drone.add_searching_time()
        if self.last_time_updates >= self._COMPUTE_FRONTIER_COOLDOWN:
            path, self.target = self.roamer.find_path(frontiers_threshold=self.frontiers_threshold, last_target=self.target)
            self.drone.set_path(path)
            if not self.first_time:
                self.last_time_updates = 0
            else:
                self.first_time_cooldown+=1
                if self.first_time_cooldown >= self._COMPUTE_FRONTIER_COOLDOWN:
                    self.first_time = False
        else:
            self.last_time_updates += 1
            return

        # if no target found
        if self.target is None:
            self.none_target_count += 1

            if self.none_target_count >= self._NONE_TARGET_FOUND_THRESHOLD:
                self.frontiers_threshold = max(self._MIN_FRONTIERS_THRESHOLD, self.frontiers_threshold - 1)
            return
        
        self.previous_searching_start_point = self.drone.drone_position
        self.count_close_previous_searching_start_point = 0
        
        self.force_going_to_target()


    @going_to_target.enter
    def on_enter_going_to_target(self):
        self.loop_count_going_to_target += 1 # increment position counter
        self.drone.reset_searching_time()
        if self.test_position_close_start_point():
            self.count_close_previous_searching_start_point += 1

        self.command = self.drone.localizer.get_control_from_path()

    def on_enter_searching_for_target(self):
        self.command = {
            "forward": 0.0,
            "lateral": 0.0,
            "rotation": 0.0,
            "grasper": 0
        }
        self.search_for_target()


    def on_enter_start_roaming(self):
        pass

    def test_position_close_start_point(self, threshold=20):
        """
        checks if the drone is close to the previous searching start point
        """
        if self.previous_searching_start_point is None: return False

        dist = self.drone.drone_position.distance(self.previous_searching_start_point)
        return dist < threshold


# TODO idea for improvement: get frontiers from other neighbor drones to save some processing time and better spread the drones to the frontiers
class Roamer:
    """
    Roamer class
    defines the roaming behavior of the drone
    """
    def __init__(self, drone: FrontierDrone, drone_map: Map, debug_mode: bool = False):
        self.frontiers : List[List[Vector2D]] = []
        self.drone : FrontierDrone = drone
        self.map : Map = drone_map
        self.debug_mode : bool = debug_mode

        self.previous_input = np.array([])
        self.previous_score = np.array([])

    
    def print_num_map(self, drone_map : Map, output_file='output.txt'):
        """
        Print the map to a file
        Debugging purposes
        """
        numeric_map = np.vectorize(lambda x: x.value)(drone_map)
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)

        with open(output_path, 'a') as file:
            np.savetxt(file, numeric_map, fmt='%3d', delimiter=', ')
            file.write('\n') 


    def find_next_unexplored_target(self, frontiers_threshold: int, last_target: Optional[Vector2D]) -> Optional[Vector2D]:
        """
        Find the closest unexplored target from the drone's current position
        It comes to finding the closest UNEXPLORED point which is next to a explored point in the map
        """
        self.drone.update_reward()
        self.drone.graph_map.update(self.drone)
        frontier_center = self.drone.graph_map.get_best_node(self.drone)

        return frontier_center


    def map_to_image(self, drone_map: Map):
        """
        returns the map as an image
        debuggin purposes
        """
        x_max_grid, y_max_grid = self.map.get_map_matrix().shape
        
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.UNEXPLORED: (0, 0, 0),
            Zone.KILL_ZONE: (255, 0, 0),
            Zone.NO_COM_ZONE: (0, 255, 0)
        }
        
        img = np.zeros((x_max_grid, y_max_grid, 3), np.uint8)
        for x in range(x_max_grid):
            for y in range(y_max_grid):
                img[x][y] = color_map[drone_map[x, y]]

        # zoom image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        return np.transpose(img, (1, 0, 2))

    def display_map_with_path(self,selected_frontiers, selected_frontiers_distance_array, frontiers_size, frontier_count, selected_frontiers_repulsion_angle_array):
        """
        Display the map with points 1 in white, points 1000 in brown, and the path in blue.
        """
        grid_map = cv2.resize(self.map.get_map_matrix(), (0, 0), fx=1/MAP_SCALE, fy=1/MAP_SCALE, interpolation=cv2.INTER_NEAREST)

        # Define color map
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.UNEXPLORED: (0, 0, 0),
            Zone.KILL_ZONE: (255, 0, 0),
            Zone.NO_COM_ZONE: (0, 255, 0)
        }

        x_max_grid, y_max_grid = grid_map.shape
        img = np.zeros((x_max_grid, y_max_grid, 3), np.uint8)

        # Assign colors to each point based on the color map
        for x in range(x_max_grid):
            for y in range(y_max_grid):
                img[x,y] = color_map[grid_map[x, y]]


        frontier_centers = []
        for frontier in selected_frontiers:
            for p in frontier:
                img[p.x//MAP_SCALE][p.y//MAP_SCALE] = (0, 255, 0)

            frontier_center = Vector2D(
                sum(point.x for point in frontier) / len(frontier),
                sum(point.y for point in frontier) / len(frontier)
            )
            frontier_centers.append(frontier_center)

        p = self.map.world_to_grid(self.drone.drone_position)
        img[p.x // MAP_SCALE, p.y // MAP_SCALE] = (255, 0, 255)

        img = cv2.resize(img, (0, 0), fx=MAP_SCALE, fy=MAP_SCALE, interpolation=cv2.INTER_NEAREST)
        plt.figure()
        plt.axis('off')
        plt.imshow(np.transpose(img, (1, 0, 2)))
        p2 = self.drone.drone_position + 80*self.drone.drone_direction_group
        p2 = self.map.world_to_grid(p2)
        plt.plot([p.x,p2.x],[p.y,p2.y], color='red', linewidth=0.5)
        for i,p in enumerate(frontier_centers):
            plt.text(p.x, p.y, str(i), horizontalalignment='center', verticalalignment='center', fontsize=8, color='red')

        with open(f"info_frontier{self.drone.identifier}.txt", 'w') as f:
            f.write(" ".join(map(str,selected_frontiers_distance_array))+'\n')
            f.write(" ".join(map(str,frontiers_size))+'\n')
            f.write(" ".join(map(str,frontier_count))+'\n')
            f.write(" ".join(map(str,selected_frontiers_repulsion_angle_array))+'\n')
            f.write(" ".join(map(str,frontier_centers)))

        # Display the image
        plt.savefig("map_debug"+str(self.drone.identifier)+".png")
        plt.close()


    # TODO: fix angle (fast shortest_path gives A* path so no indication on direction)
    def get_path_length(self, target : Optional[Vector2D]) -> Tuple[float,float,float]:
        """
        Get the length of the path to the target
        params:
            - target: the target
        """
        if target is None: return np.inf, 1, 1
        path, next_waypoint = self.map.shortest_path(self.drone.drone_position, self.map.grid_to_world(target), fast=True)

        if path is None or len(path) <= 1:
            return np.inf, 1, 1
        
        path_length = 0
        for k in range(len(path)-1):
            path_length += path[k].distance(path[k+1])
        
        waypoint_direction : Vector2D = next_waypoint - self.drone.drone_position
        waypoint_direction.normalize()


        direction_angle = self.drone.drone_angle - waypoint_direction.get_angle(Vector2D(1,0))
        direction_angle = abs(normalize_angle(direction_angle))/np.pi

        repulsion_angle = 1
        repulsion_direction = self.drone.drone_direction_group
        if repulsion_direction.norm()>0:
            repulsion_angle = abs(normalize_angle(waypoint_direction.get_angle(repulsion_direction)))/np.pi

        return path_length, repulsion_angle, direction_angle



    def find_path(self, frontiers_threshold: int, last_target:Optional[Vector2D]):
        """
        Find the path to the target
        params
            - sampling_rate: the sampling rate of the path (in order to reduce the number of points)
            - last_target: the last selected frontier
        """
        target = self.find_next_unexplored_target(frontiers_threshold, last_target)

        if target is None:
            return [], None
        
        if self.drone.drone_position.distance(self.drone.map.grid_to_world(target)) < 50:
            return [], None

        path = self.map.shortest_path(self.drone.drone_position, target)[0]

        if path is None:
            return [], None

        return path, target