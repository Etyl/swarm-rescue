from statemachine import StateMachine, State
from solutions.mapper.mapper import Map
from solutions.mapper.mapper import Zone
from spg_overlay.entities.drone_abstract import DroneAbstract
import numpy as np
from enum import Enum
import pyastar2d
import time
import threading
import os
import cv2
import asyncio
import math
from collections import deque

class RoamerController(StateMachine):
    start_roaming = State('Start Roaming', initial=True)
    searching_for_target = State('Searching for target')
    going_to_target = State('Going to target')

    force_transition = start_roaming.to(searching_for_target, on='before_searching_for_target')
    force_transition2 = searching_for_target.to(going_to_target, on='before_going_to_target')

    cycle = (
        start_roaming.to(searching_for_target, cond="drone_position_valid") |
        searching_for_target.to(going_to_target) |
        # going_to_target.to(searching_for_target, cond="target_discorvered") 
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
        self.roamer = Roamer(drone, map)

        super(RoamerController, self).__init__()
    
    def check_target_reached(self):
        """
        checks if the drone has reached the waypoint
        """

        dist = np.linalg.norm(self.drone.get_position() - self.target)
        if len(self.drone.path) == 0: return dist < 150

        v1 = self.target- self.drone.get_position()
        v2 = np.array(self.drone.path[-1]) - np.array(self.map.grid_to_world(self.target))
        turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        return dist < 200 + (1+turning_angle)*20

    def drone_position_valid(self):
        return self.drone.get_position() is not None and not np.isnan(self.drone.get_position()).any()

    def target_discorvered(self):
        return self.map[self.target] != Zone.INEXPLORED

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        return f"Running {event} from {source.id} to {target.id}{message}"
    
    def before_going_to_target(self):
        print(f"Going to target {self.target}")

        # Compute path to target
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True
    
    async def search_for_target(self):
        await asyncio.sleep(1)
        self.drone.path, self.target = self.roamer.find_path()
        self.force_transition2()

    def before_searching_for_target(self):
        print("Searching for target")

        # Compute path to target
        # search_thread = threading.Thread(target=self.search_for_target)
        # search_thread.start()

        # asyncio.run(self.search_for_target())


    @going_to_target.enter
    def on_enter_going_to_target(self):
        self.command = self.drone.get_control_from_path(self.drone.get_position())

    def on_enter_searching_for_target(self):
        print("Entering searching for target state")
        asyncio.run(self.search_for_target())

    def on_enter_start_roaming(self):
        print("Entering start roaming state")

class Roamer:
    def __init__(self, drone: DroneAbstract, map: Map):
        self.drone = drone
        self.map = map
        self.map_matrix = map.map
    
    def print_num_map(self, map, output_file='output.txt'):
        numeric_map = np.vectorize(lambda x: x.value)(map)
        
        # Save to file with alignment
        script_dir = os.path.dirname(os.path.abspath(__file__))
        output_path = os.path.join(script_dir, output_file)

        with open(output_path, 'a') as file:
            np.savetxt(file, numeric_map, fmt='%3d', delimiter=', ')
            file.write('\n') 
    
    def find_next_unexeplored_target(self):
        """
        Find the closest unexplored target from the drone's curretn position
        It comes to finding the closest INEXPLORED point which is next to a explored point in the map
        """
        print("[Roamer] Finding next unexplored target")
        rows, cols = self.map_matrix.shape
        print("[Roamer] Drone position : ", self.drone.get_position())
        current_row, current_col = self.map.world_to_grid(self.drone.get_position())
        print("[Roamer] Current position : ", current_row, current_col)

        map_matrix_copy = self.map_matrix.copy()

        self.print_num_map(map_matrix_copy)
        
        def is_valid_move(row, col):
            return 0 <= row < rows and 0 <= col < cols and self.map[row, col] != Zone.OBSTACLE and self.map[row, col] != Zone.RESCUE_CENTER

        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Helper function to check if a cell has a neighbor with Zone value >= 0
        def has_valid_neighbors(row, col, n, r):
            empty_neighbors = 0

            for dr in range(-r, r+1):
                for dc in range(-r, r+1):
                    new_row, new_col = row + dr, col + dc

                    if (
                        0 <= new_row < rows and 0 <= new_col < cols
                        and self.map[new_row, new_col] == Zone.EMPTY
                        and (dr, dc) != (0, 0)  # Exclude the current point
                    ):
                        empty_neighbors += 1

                        if empty_neighbors >= n:
                            return True

            return False
        
        def is_surrounding_empty(row, col, radius):
            for i in range(-radius, radius + 1):
                for j in range(-radius, radius + 1):
                    if not is_valid_move(row + i, col + j) or self.map[row + i, col + j] != Zone.EMPTY:
                        return False
            return True

        # BFS to find the closest unexplored point with a valid neighbor
        found_point = None
        queue = deque([(current_row, current_col)])
        visited = set()

        while queue:
            current_row, current_col = queue.popleft()
            
            if (len(queue) % 100) == 0:
                self.print_num_map(map_matrix_copy)
                self.display_map(map_matrix_copy)

            if (
                self.map[current_row, current_col] == Zone.INEXPLORED
                and has_valid_neighbors(current_row, current_col, 10, 5)
            ):
                print(f"Found unexplored target at {current_row, current_col}")
                found_point = np.array([current_row, current_col])
                queue = None
                break

            map_matrix_copy[current_row, current_col] = Zone.RESCUE_CENTER

            visited.add((current_row, current_col))

            for dr, dc in moves:
                new_row, new_col = current_row + dr, current_col + dc
                if is_valid_move(new_row, new_col) and (new_row, new_col) not in visited:
                    queue.append((new_row, new_col))

        print("[Roamer] Found next unexplored target : ", found_point)
        return found_point

    def map_to_image(self, map):
        x_max_grid, y_max_grid = self.map_matrix.shape
        
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

    def display_map(self, map):
        img = self.map_to_image(map)
        # TODO: fix this
        # parce que openCV ne marche pas en multithread
        cv2.imshow("map2", img)
        cv2.waitKey(1)

    def convert_matrix_for_astar(self, matrix):
        INF = 1000

        conversion_dict = {
            Zone.INEXPLORED: INF,
            Zone.OBSTACLE: INF,
            Zone.RESCUE_CENTER: INF,
            Zone.WOUNDED: 1,
            Zone.EMPTY: 1,
        }

        converted_matrix = np.vectorize(lambda x: conversion_dict[x])(matrix)
        return converted_matrix.astype(np.float32)

    def find_path(self, sampling_rate: int = 1):
        """
        Find the path to the target
        """
        target = self.find_next_unexeplored_target()

        matrix_astar = self.convert_matrix_for_astar(self.map_matrix)
        drone_position_grid = self.map.world_to_grid(self.drone.get_position())

        path = pyastar2d.astar_path(matrix_astar, tuple(drone_position_grid), tuple(target), allow_diagonal=True)
        path_sampled = path[::sampling_rate]

        # convert path to world coordinates
        path_sampled = np.array([self.map.grid_to_world(pos) for pos in path_sampled])
        print("[Roamer] Path found : ", path_sampled)
        return path_sampled.tolist(), target
    


    
