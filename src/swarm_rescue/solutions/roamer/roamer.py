"""
Roamer module
implements roaming behavior of the drone
"""
from statemachine import StateMachine, State
from solutions.mapper.mapper import Map
from solutions.mapper.mapper import Zone
from solutions.roamer.frontier_explorer import FrontierExplorer
from spg_overlay.entities.drone_abstract import DroneAbstract
from enum import Enum
from scipy.ndimage import binary_dilation
from scipy.ndimage import convolve
from solutions.pathfinder.pathfinder import *

import math
import numpy as np
import os
import cv2

from solutions.utils import normalize_angle

# OTHER IMPL - ASYNC
# import asyncio
# import threading

FRONTIER_SELECTION_SIZE = 10

class RoamerController(StateMachine):

    # maximum number of times the drone can be in the same position
    # i.e the maximum number of times the check_target_reached function can return False
    _LOOP_COUNT_GOING_TO_TARGET_THRESHOLD = 100

    # maximum number of times the drone can be close to the previous searching start point
    # i.e the maximum number of times the test_position_close_start_point function can return True
    _COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD = 50

    # maximum number of times the drone can't find a target
    # i.e the maximum number of times the search_for_target function can return None
    _NONE_TARGET_FOUND_THRESHOLD = 10

    # the thickness of the walls in the map when the path is computed (in order to allow a smoother path)
    _WALL_THICKENING = 3

    # the sampling rate of the path (in order to reduce the number of points)
    _PATH_SAMPLING_RATE = 1

    # the number of points required in a frontier
    _FRONTIERS_THRESHOLD = 6

    # the cooldown for computing the frontier
    _COMPUTE_FRONTIER_COOLDOWN = 50

    start_roaming = State('Start Roaming', initial=True)
    searching_for_target = State('Searching for target')
    going_to_target = State('Going to target')

    force_searching_for_target = start_roaming.to(searching_for_target, on='before_searching_for_target')
    force_going_to_target = searching_for_target.to(going_to_target, on='before_going_to_target')

    cycle = (
        start_roaming.to(searching_for_target, cond="drone_position_valid") |

        searching_for_target.to(going_to_target) |

        going_to_target.to(searching_for_target, cond="no_path_to_target") |
        going_to_target.to(searching_for_target, cond="check_target_reached") |
        going_to_target.to(going_to_target)
    )

    def __init__(self, drone: DroneAbstract, map: Map, debug_mode: bool = False):
        self.drone = drone
        self.map = map
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.debug_mode = debug_mode
        self.roamer = Roamer(drone, map, debug_mode)

        self.loop_count_going_to_target = 0

        self.target = None
        self.none_target_count = 0
        
        # numbers of points required in a frontier
        # the idea is to decrease this number if the drone is stuck at some point
        # because it would mean that the drone has already explored a lot of the map
        # and that the remaining unexplored areas are small
        self.frontiers_threshold = self._FRONTIERS_THRESHOLD

        self.previous_searching_start_point = None
        self.count_close_previous_searching_start_point = 0

        self.waiting_time = 0
        self.last_time_updates = self._COMPUTE_FRONTIER_COOLDOWN
        self.first_time = True
        self.first_time_cooldown = 0

        self.frontiers = None

        super(RoamerController, self).__init__()

    def no_path_to_target(self):
        # TODO change implementation, use velocity (check if stuck)
        # right now, we only increment a counter each time the check is called
        # if the counter reaches a certain threshold, we restart the search
        if self.loop_count_going_to_target >= self._LOOP_COUNT_GOING_TO_TARGET_THRESHOLD: return True

        if self.count_close_previous_searching_start_point >= self._COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD: return True

        # in case the drone has ended its path
        if self.drone.nextWaypoint is None: return True

        # TODO fix this
        if self.target is None or self.target==0: return True
    
    def check_target_reached(self):
        """
        checks if the drone has reached the waypoint
        """

        dist = np.linalg.norm(self.drone.get_position() - self.target)
        if len(self.drone.path) == 0: return dist < 20

        v1 = self.target- self.drone.get_position()

        v2 = np.array(self.drone.path[-1]) - np.array(self.map.grid_to_world(self.target))
        
        if np.linalg.norm(v1) == 0 or np.linalg.norm(v2) == 0: 
            turning_angle = 0
        else: 
            turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        return dist < 20 + (1+turning_angle)*20

    def drone_position_valid(self):
        return self.drone.get_position() is not None and not np.isnan(self.drone.get_position()).any()

    def target_discorvered(self):
        return (self.map[self.target] != Zone.INEXPLORED or self.drone.nextWaypoint is None)

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        return f"Running {event} from {source.id} to {target.id}{message}"
    
    def before_going_to_target(self):
        if self.debug_mode: print(f"Going to target {self.target}")

        # TODO change implementation
        self.loop_count_going_to_target = 0 # reset position counter

        self.none_target_count = 0 # reset none target counter

        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True
    
    def search_for_target(self):
        # OTHER IMPL - ASYNC        
        # await asyncio.sleep(1)
        # END OTHER IMPL - ASYNC

        if self.last_time_updates >= self._COMPUTE_FRONTIER_COOLDOWN:
            self.drone.path, self.target = self.roamer.find_path(frontiers_threshold=self.frontiers_threshold)
            if not self.first_time:
                self.last_time_updates = 0
            else:
                self.first_time_cooldown+=1
                if self.first_time_cooldown >= self._COMPUTE_FRONTIER_COOLDOWN:
                    self.first_time = False
        else:
            self.last_time_updates += 1
            return
        # if need to wait
        if self.target == -1:
            self.command = {"forward": 0.0,
                            "lateral": 0.0,
                            "rotation": 0.1,
                            "grasper": 0}
            return

        # if no target found
        if self.target is None:
            if self.debug_mode: print("[Roamer] No path to target found")
            self.none_target_count += 1

            if self.none_target_count >= self._NONE_TARGET_FOUND_THRESHOLD:
                self.frontiers_threshold = max(1, self.frontiers_threshold - 1)
            return
        
        # if target is too close
        if self.target == 0:
            if self.debug_mode: print("[Roamer] Target too close")
            self.command = {"forward": 0.0,
                            "lateral": 0.0,
                            "rotation": 0.0,
                            "grasper": 0}
            return
        
        self.previous_searching_start_point = self.drone.get_position()
        self.count_close_previous_searching_start_point = 0
        
        self.force_going_to_target()

    def before_searching_for_target(self):
        if self.debug_mode: print("Searching for target")

        # OTHER IMPL - ASYNC        
        # search_thread = threading.Thread(target=self.search_for_target)
        # search_thread.start()
        # asyncio.run(self.search_for_target())
        # END OTHER IMPL - ASYNC


    @going_to_target.enter
    def on_enter_going_to_target(self):
        # TODO change implementation
        self.loop_count_going_to_target += 1 # increment position counter

        if self.test_position_close_start_point():
            self.count_close_previous_searching_start_point += 1

        self.command = self.drone.get_control_from_path(self.drone.get_position())

    def on_enter_searching_for_target(self):
        if self.debug_mode: print("Entering searching for target state")
        self.search_for_target()

        # OTHER IMPL - ASYNC        
        # asyncio.run(self.search_for_target())
        # END OTHER IMPL - ASYNC

    def on_enter_start_roaming(self):
        if self.debug_mode: print("Entering start roaming state")

    def test_position_close_start_point(self, threshold=20):
        """
        checks if the drone is close to the previous searching start point
        """
        if self.previous_searching_start_point is None: return False

        dist = np.linalg.norm(self.drone.get_position() - self.previous_searching_start_point)
        return dist < threshold

class Roamer:
    """
    Roamer class
    defines the roaming behavior of the drone
    """
    def __init__(self, drone: DroneAbstract, map: Map, debug_mode: bool = False):
        self.drone = drone
        self.map = map
        self.debug_mode = debug_mode
    
    def print_num_map(self, map, output_file='output.txt'):
        """
        Print the map to a file
        Debugging purposes
        """
        numeric_map = np.vectorize(lambda x: x.value)(map)        
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)

        with open(output_path, 'a') as file:
            np.savetxt(file, numeric_map, fmt='%3d', delimiter=', ')
            file.write('\n') 
    
    def find_next_unexplored_target(self, frontiers_threshold):
        """
        Find the closest unexplored target from the drone's curretn position
        It comes to finding the closest INEXPLORED point which is next to a explored point in the map
        """

        map_matrix_copy = self.map.get_map_matrix().copy() # copy map (to not modify the original)
  
        map_matrix_copy = np.vectorize(lambda zone: zone.value)(map_matrix_copy) # convert to int (for the Frontier Explorer algorithms)
        drone_position = self.drone.get_position()

        if self.debug_mode: print("[Roamer] Drone position : ", drone_position, self.map.world_to_grid(drone_position))
        
        drone_position_grid = self.map.world_to_grid(drone_position)
        fd = FrontierExplorer(map_matrix_copy, drone_position_grid, frontiers_threshold)
        frontiers = fd.getFrontiers()
        self.frontiers = frontiers
        
        frontier_count = fd.getFrontiersCount()

        if not frontiers:
            return None  # No frontiers found

        # Find the frontier with the closest center to the robot
        best_distance = float('inf')
        best_frontier_idx = 0
        best_count = 0

        max_distance = np.inf
        selected_frontiers_id = []
        selected_frontiers_distance = []
        selected_frontiers_repulsion_angle = []
        
        # select the closest frontiers
        for idx, frontier in enumerate(frontiers):
            
            frontier_center = (
                sum(point[0] for point in frontier) / len(frontier),
                sum(point[1] for point in frontier) / len(frontier)
            )
            
            distance = np.linalg.norm(np.array(frontier_center) - np.array(drone_position_grid))
            
            if len(selected_frontiers_id) < FRONTIER_SELECTION_SIZE :
                selected_frontiers_id.append(idx)
                selected_frontiers_distance.append(distance)
                continue
            elif distance < max(selected_frontiers_distance):
                max_idx = selected_frontiers_distance.index(max(selected_frontiers_distance))
                selected_frontiers_id[max_idx] = idx
                selected_frontiers_distance[max_idx] = distance
                continue
        
        # calculate the path length for each selected frontier
        for id,idx in enumerate(selected_frontiers_id):
            frontier = frontiers[idx]
            frontier_center = (
                sum(point[0] for point in frontier) / len(frontier),
                sum(point[1] for point in frontier) / len(frontier)
            )
            distance, repulsion_angle = self.get_path_length(frontier_center)
            selected_frontiers_distance[id] = distance
            selected_frontiers_repulsion_angle.append(repulsion_angle)

        # parameters
        frontiers = [frontiers[idx] for idx in selected_frontiers_id]
        frontier_count = np.array([frontier_count[idx] for idx in selected_frontiers_id])
        selected_frontiers_distance = np.array(selected_frontiers_distance)
        selected_frontiers_repulsion_angle = np.array(selected_frontiers_repulsion_angle)
        frontiers_size = np.array([len(frontier) for frontier in frontiers])
        
        # normalize
        frontier_distance_noInf = [x for x in selected_frontiers_distance if x != np.inf]
        if len(frontier_distance_noInf) == 0: return None
        selected_frontiers_distance = selected_frontiers_distance / max(frontier_distance_noInf)

        frontier_size_max = np.max(frontiers_size)
        frontiers_size = 0 if frontier_size_max == 0 else frontiers_size / frontier_size_max

        frontier_count_max = np.max(frontier_count)
        frontier_count = 0 if frontier_count_max == 0 else frontier_count / frontier_count_max

        # score (the higher the better)
        # TODO : optimize
        score = 2*(1-selected_frontiers_distance) + frontiers_size + frontier_count + 4*(1-selected_frontiers_repulsion_angle)

        # select the best frontier
        best_frontier_idx = np.argmax(score)

        # Return the center and the points of the chosen frontier
        chosen_frontier = frontiers[best_frontier_idx]
        chosen_frontier_center = (
            int(sum(point[0] for point in chosen_frontier) / len(chosen_frontier)),
            int(sum(point[1] for point in chosen_frontier) / len(chosen_frontier))
        )

        # print(f"best_count: {best_count} - best_distance: {best_distance}")

        if self.debug_mode: print("[Roamer] Found point : ", chosen_frontier_center)

        return chosen_frontier_center


    def map_to_image(self, map):
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
            Zone.INEXPLORED: (0, 0, 0)
        }
        
        img = np.zeros((x_max_grid, y_max_grid, 3), np.uint8)
        for x in range(x_max_grid):
            for y in range(y_max_grid):
                img[x][y] = color_map[map[x, y]]

        # zoom image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        return np.transpose(img, (1, 0, 2))

    def display_map_with_path(self, grid_map, path, target, id):
        """
        Display the map with points 1 in white, points 1000 in brown, and the path in blue.
        """
        x_max_grid, y_max_grid = self.map.get_map_matrix().shape

        # Define color map
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.INEXPLORED: (0, 0, 0),
            1: (255, 255, 255),  # Color for points with value 1 (white)
            1000: (139, 69, 19)  # Color for points with value 1000 (brown)
        }

        img = np.zeros((x_max_grid, y_max_grid, 3), np.uint8)

        # Assign colors to each point based on the color map
        for x in range(x_max_grid):
            for y in range(y_max_grid):
                img[x][y] = color_map[grid_map[x, y]]
                
        for frontier in self.frontiers:
            for x, y in frontier:
                img[x][y] = (0, 255, 0)

        if path is not None:
            for pos in path:
                x,y = self.map.world_to_grid(pos)
                img[x][y] = (255, 0, 0)
        
        img[target[0]][target[1]] = (0 , 0, 255)
        img[target[0]+1][target[1]] = (0 , 0, 255)
        img[target[0]][target[1]+1] = (0 , 0, 255)
        img[target[0]][target[1]-1] = (0 , 0, 255)
        img[target[0]-1][target[1]] = (0 , 0, 255)

        # Convert coordinates to integers and assign blue color to the path
        # for coord in path:
        #     x, y = map(int, coord)
        #     img[x, y] = (0, 0, 255)

        # Zoom image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

        # Display the image
        cv2.imshow("map_debug" + str(id), np.transpose(img, (1, 0, 2)))
        cv2.waitKey(1)


    def convert_matrix_for_astar(self, matrix):
        """
        Convert the map matrix to a matrix that can be used by the A* algorithm
        params:
            - matrix: the map matrix with the Zone values
        """
        INF = 1000 # INF = 1000 c'est bien connu

        conversion_dict = {
            Zone.INEXPLORED: INF,
            Zone.OBSTACLE: INF,
            Zone.RESCUE_CENTER: INF,
            Zone.WOUNDED: 1,
            Zone.EMPTY: 1,
        }

        converted_matrix = np.vectorize(lambda x: conversion_dict[x])(matrix)
        return converted_matrix.astype(np.float32)
    

    def convert_matrix_for_pathfinder(self, matrix):
        """
        Convert the map matrix to a matrix that can be used by the A* algorithm
        params:
            - matrix: the map matrix with the Zone values
        """
        INF = 1000 # INF = 1000 c'est bien connu

        conversion_dict = {
            Zone.INEXPLORED: 0,
            Zone.OBSTACLE: 0,
            Zone.RESCUE_CENTER: 0,
            Zone.WOUNDED: 1,
            Zone.EMPTY: 1,
        }

        converted_matrix = np.vectorize(lambda x: conversion_dict[x])(matrix)
        return converted_matrix.astype(np.float32)
    
    def thicken_walls(self, matrix, n):
        """
        Thicken the walls in the map
        in order to allow a smoother path
        """
        new_matrix = np.copy(matrix)
        rows, cols = matrix.shape

        for i in range(rows):
            for j in range(cols):
                if matrix[i, j] == Zone.OBSTACLE:
                    for x in range(max(0, i - n), min(rows, i + n + 1)):
                        for y in range(max(0, j - n), min(cols, j + n + 1)):
                            if matrix[x, y] == Zone.EMPTY:
                                new_matrix[x, y] = Zone.OBSTACLE

        return new_matrix

    def find_empty_point_near_wall(self, drone_position, grid_map, p):
        x_max, y_max = grid_map.shape

        # Définir la liste des voisins pour un point donné
        def neighbors(point):
            x, y = point
            return [(x + i, y + j) for i in [-1, 0, 1] for j in [-1, 0, 1]
                    if 0 <= x + i < x_max and 0 <= y + j < y_max and (i != 0 or j != 0)]

        # Calculer la distance entre deux points
        def distance(point1, point2):
            return np.sqrt((point1[0] - point2[0])**2 + (point1[1] - point2[1])**2)

        # Parcourir les points dans l'ordre croissant de distance depuis le drone
        for d in range(1, max(x_max, y_max)):
            for i in range(-d, d + 1):
                for j in range(-d, d + 1):
                    current_point = (drone_position[0] + i, drone_position[1] + j)

                    # Vérifier si le point est à une distance minimale d'une case mur
                    if (0 <= current_point[0] < x_max and 0 <= current_point[1] < y_max
                            and grid_map[current_point] == 1):
                        if all(grid_map[neighbor] != 1000 for neighbor in neighbors(current_point)):
                            # Trouvé un point vide éloigné d'au moins p cases d'une case mur
                            return current_point

        # Si aucun point n'est trouvé, retourner None
        return None
    
    def get_path_length(self, target):
        """
        Get the length of the path to the target
        params:
            - target: the target
        """
        if target is None: return np.inf, 1
        path = self.map.shortest_path(self.drone.get_position(), self.map.grid_to_world(target))

        if path is None or len(path) <= 1:
            return np.inf, 1
        
        path_length = np.sum(np.linalg.norm(np.diff(path, axis=0), axis=1))

        def compute_angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        repulsion = self.drone.repulsion
        if np.linalg.norm(repulsion)>0 : repulsion /= np.linalg.norm(repulsion)
        
        nextWaypoint = path[-2]
        waypointDirection = nextWaypoint - self.drone.get_position()
        waypointDirection /= np.linalg.norm(waypointDirection)

        angleWaypoint = compute_angle(waypointDirection, np.array([1,0]))
        angleWaypoint = normalize_angle(angleWaypoint+self.drone.get_angle())
        angleRepulsion = compute_angle(repulsion, np.array([1,0]))
        angle = normalize_angle(angleRepulsion-angleWaypoint)

        return path_length, abs(angle)/np.pi



    def find_path(self, frontiers_threshold):
        """
        Find the path to the target
        params
            - sampling_rate: the sampling rate of the path (in order to reduce the number of points)
        """
        target = self.find_next_unexplored_target(frontiers_threshold)

        # TODO change implementation
        if target is None:
            return [], None
        
        if target == -1:
            return [], -1
        
        if np.linalg.norm(self.drone.get_position() - self.drone.map.grid_to_world(target)) < 50:
            return [], 0

        path = self.map.shortest_path(self.drone.get_position(), self.map.grid_to_world(target))
        
        if self.debug_mode: 
            print("[Roamer] Path found : ", path)
            self.display_map_with_path(self.map.get_map_matrix(), path, target, 5)

        # TODO change implementation
        if path is None:
            return [], None

        return path, target