from statemachine import StateMachine, State
from solutions.mapper.mapper import Map
from solutions.mapper.mapper import Zone
from solutions.roamer.frontier_explorer import FrontierExplorer
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
        self.pos_count = 0
        self.target = None

        super(RoamerController, self).__init__()
    
    def check_target_reached(self):
        """
        checks if the drone has reached the waypoint
        """

        if self.pos_count >= 100: return True
        if self.drone.nextWaypoint is None: return True
        if self.target is None: return False

        dist = np.linalg.norm(self.drone.get_position() - self.target)
        if len(self.drone.path) == 0: return dist < 20

        v1 = self.target- self.drone.get_position()
        v2 = np.array(self.drone.path[-1]) - np.array(self.map.grid_to_world(self.target))
        turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        return dist < 20 + (1+turning_angle)*20 or self.drone.nextWaypoint is None

    def drone_position_valid(self):
        return self.drone.get_position() is not None and not np.isnan(self.drone.get_position()).any()

    def target_discorvered(self):
        print(self.drone.nextWaypoint)
        return (self.map[self.target] != Zone.INEXPLORED or self.drone.nextWaypoint is None)

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        return f"Running {event} from {source.id} to {target.id}{message}"
    
    def before_going_to_target(self):
        print(f"Going to target {self.target}")
        self.pos_count = 0

        # Compute path to target
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True
    
    def search_for_target(self):
        # await asyncio.sleep(1)
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
        self.pos_count += 1 
        self.command = self.drone.get_control_from_path(self.drone.get_position())

    def on_enter_searching_for_target(self):
        print("Entering searching for target state")
        # asyncio.run(self.search_for_target())
        self.search_for_target()

    def on_enter_start_roaming(self):
        print("Entering start roaming state")

class Roamer:
    def __init__(self, drone: DroneAbstract, map: Map):
        self.drone = drone
        self.map = map
        self.map_matrix = map.map
        self.frontier_explorer = FrontierExplorer(self.map_matrix, drone)
    
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
        
        map_matrix_copy = self.map_matrix.copy() # copy map
        map_matrix_copy = np.vectorize(lambda zone: zone.value)(map_matrix_copy) # convert to int
        drone_position = self.drone.get_position()
        print("[Roamer] Drone position : ", drone_position, self.map.world_to_grid(drone_position))
        fd = FrontierExplorer(map_matrix_copy, self.map.world_to_grid(drone_position))

        found_point = fd.getClosestFrontier()
        print("[Roamer] Found point : ", found_point)
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
        
        if target is None:
            return [], None

        matrix_astar = self.convert_matrix_for_astar(self.map_matrix)
        drone_position_grid = self.map.world_to_grid(self.drone.get_position())

        path = pyastar2d.astar_path(matrix_astar, tuple(drone_position_grid), tuple(target), allow_diagonal=True)
        path_sampled = path[:-int(len(path) * 0.2)]
        path_sampled = path_sampled[::sampling_rate]

        # convert path to world coordinates
        path_sampled = np.array([self.map.grid_to_world(pos) for pos in path_sampled])
        print("[Roamer] Path found : ", path_sampled)
        path_list = path_sampled.tolist()
        path_list.reverse()
        return path_list, target
    


    
