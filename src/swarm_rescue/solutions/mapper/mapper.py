from __future__ import annotations

import numpy as np
import cv2
from typing import List, TYPE_CHECKING, Optional, Dict, Tuple
import matplotlib.pyplot as plt

from solutions.utils.types import Vector2D  # type: ignore
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor # type: ignore
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_SEMANTIC_SENSOR  # type: ignore
from solutions.pathfinder.pathfinder import pathfinder, pathfinder_fast # type: ignore
from solutions.mapper.utils import display_grid, Grid, MerkleTree # type: ignore

if TYPE_CHECKING: # type: ignore
    from solutions.frontier_drone import FrontierDrone # type: ignore

LIDAR_MAPPER_RANGE = 0.60*MAX_RANGE_LIDAR_SENSOR
EVERY_N = 1
LIDAR_DIST_CLIP = 0.0
EMPTY_ZONE_VALUE = -0.4
OBSTACLE_ZONE_VALUE = 2.0
FREE_ZONE_VALUE = -8
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40
CONFIDENCE_VALUE = 500
CONFIDENCE_THRESHOLD = 1000
CONFIDENCE_THRESHOLD_MIN = 0
KILL_ZONE_SIZE = 15
SAFE_ZONE_SIZE = 30
KILL_ZONE_INCREFMENT = 0.2

DRONE_SIZE_RADIUS = 12

class Zone:
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    UNEXPLORED = -1
    KILL_ZONE = 4
    NO_COM_ZONE = 5

class Map:
    def __init__(self, area_world: List[int], resolution: int, identifier:int, debug_mode: bool=False) -> None:
        self.resolution: int = resolution
        self.debug_mode: bool = debug_mode
        self.identifier: int = identifier

        self.width : int = int(area_world[0] / self.resolution + 0.5)
        self.height : int = int(area_world[1] / self.resolution + 0.5)

        self.occupancy_grid : Grid = Grid(area_world, resolution)
        self.kill_zone : Grid = Grid(area_world, resolution)
        self.binary_occupancy_grid : np.ndarray = np.zeros((self.width, self.height)).astype(np.uint8)
        self.confidence_grid : Grid = Grid(area_world, resolution)
        self.confidence_grid_downsampled : Grid = Grid(area_world, resolution * 2)
        self.confidence_wall_map : np.ndarray = np.zeros((self.width, self.height)).astype(np.float64)
        self.merkle_tree: MerkleTree = MerkleTree(self.confidence_grid, self.occupancy_grid)
        self.kill_zone_merkle_tree: MerkleTree = MerkleTree(self.confidence_grid, self.kill_zone)

        self.map : np.ndarray = np.full((self.width, self.height), Zone.UNEXPLORED)

        self.rescue_center : Optional[Vector2D] = None
        self.kill_zones : List[Vector2D] = {}
        self.no_gps_zones : List[Vector2D] = []

    @property
    def exploration_score(self) -> float:
        return np.count_nonzero(self.map-Zone.UNEXPLORED) / (self.height * self.width)

    def has_wall(self, start: Vector2D, end:Vector2D) -> bool:
        n = int(round((end-start).norm())+1)
        for i in range(n+1):
            p = start + (i/n)*(end-start)
            x = int(round(p.x))
            y = int(round(p.y))
            if not (0<=x<self.width and 0<=y<self.height):
                continue
            if self.map[x, y] == Zone.OBSTACLE or self.map[x,y] == Zone.UNEXPLORED or self.map[x,y] == Zone.KILL_ZONE:
                return True
        return False

    def is_reachable(self, position:Vector2D, target: Vector2D) -> bool:
        d = (target - position).norm() - 2*DRONE_SIZE_RADIUS
        target = position + d * (target-position).normalize()
        position = self.world_to_grid(position)
        target = self.world_to_grid(target)
        normal = (target - position).normalize().rotate(np.pi / 2)
        for i in range(-1,2):
            start = position +  i*(DRONE_SIZE_RADIUS/self.resolution)*normal
            end = target + i*(DRONE_SIZE_RADIUS/self.resolution)*normal
            if self.has_wall(start,end):
                return False
        return True
    
    def update_confidence_grid(self, pose, lidar, drone: FrontierDrone):
        """
        Update confidence grid with drone's semantic sensor data
        """
        lidar_dist = lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = lidar.ray_angles[::EVERY_N].copy()

        lidar_dist = np.minimum(np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0), LIDAR_MAPPER_RANGE)

        world_points = np.column_stack((pose.position[0] + np.multiply(lidar_dist, np.cos(lidar_angles + pose.orientation)), pose.position[1] + np.multiply(lidar_dist, np.sin(lidar_angles + pose.orientation))))

        if not drone.gps_disabled:
            self.confidence_grid_downsampled.add_value_along_lines_confidence(pose.position[0], pose.position[1], world_points[:,0].astype(np.float32), world_points[:,1].astype(np.float32), CONFIDENCE_VALUE)
        else:
            self.confidence_grid_downsampled.add_value_along_lines_confidence(pose.position[0], pose.position[1], world_points[:,0].astype(np.float32), world_points[:,1].astype(np.float32), CONFIDENCE_VALUE/2)

        self.confidence_grid_downsampled.set_grid(np.clip(self.confidence_grid_downsampled.get_grid(), 0, CONFIDENCE_THRESHOLD))
        self.confidence_grid.set_grid(cv2.resize(self.confidence_grid_downsampled.get_grid(), (self.height, self.width), interpolation=cv2.INTER_LINEAR_EXACT))

    def update_occupancy_grid(self, pose, lidar, drone: FrontierDrone) -> None:
        """
        Update occupancy grid with drone's semantic sensor data
        """
        drone_pos: Vector2D = drone.drone_position
        drone_angle: float = drone.drone_angle

        # Save values at the boundaries
        boundary_mask = np.logical_or(self.occupancy_grid.get_grid() == THRESHOLD_MIN, self.occupancy_grid.get_grid() == THRESHOLD_MAX)
        buffer = self.occupancy_grid.get_grid().copy()

        lidar_dist = lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = lidar.ray_angles[::EVERY_N].copy()

        lidar_dist_clip = np.minimum(np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0), LIDAR_MAPPER_RANGE)

        world_points_free = np.column_stack((
            drone_pos.x + np.multiply(lidar_dist_clip, np.cos(lidar_angles + drone_angle)),
            drone_pos.y + np.multiply(lidar_dist_clip, np.sin(lidar_angles + drone_angle))
        ))

        self.occupancy_grid.add_value_along_lines(pose.position[0], pose.position[1], world_points_free[:,0].astype(np.float32), world_points_free[:,1].astype(np.float32), EMPTY_ZONE_VALUE)

        detection_semantic = drone.semantic_values()
        angles_available_arr = np.ones(181).astype(bool)
        for data in detection_semantic:
            if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                continue
            i = int(round(180 * ((np.pi + data.angle) / (2 * np.pi))))
            for k in range((i - 3) % 181, (i + 4) % 181):
                angles_available_arr[k] = False

        mask = np.logical_and(angles_available_arr,lidar_dist < LIDAR_MAPPER_RANGE)
        lidar_dist_hit = lidar_dist[mask]
        lidar_angles_hit = lidar_angles[mask]

        world_points_hit = np.column_stack((pose.position[0] + np.multiply(lidar_dist_hit, np.cos(lidar_angles_hit + pose.orientation)), pose.position[1] + np.multiply(lidar_dist_hit, np.sin(lidar_angles_hit + pose.orientation))))

        self.occupancy_grid.set_grid(np.where(boundary_mask, buffer, self.occupancy_grid.get_grid()))
        self.occupancy_grid.add_points_obstacle(world_points_hit[:,0].astype(np.float32), world_points_hit[:,1].astype(np.float32), OBSTACLE_ZONE_VALUE)

        # Mark the current position of the drone as free
        self.occupancy_grid.add_point(int(pose.position[0]), int(pose.position[1]), FREE_ZONE_VALUE)

        self.occupancy_grid.set_grid(np.clip(self.occupancy_grid.get_grid(), THRESHOLD_MIN, THRESHOLD_MAX))
        self.binary_occupancy_grid = np.where(self.occupancy_grid.get_grid() > 0, 1, -1)

        #self.filter_occupancy_grid()

    
    def filter_occupancy_grid(self):
        """
        Filter the occupancy grid with a median filter
        """
        #self.binary_occupancy_grid = cv2.erode(self.binary_occupancy_grid, np.ones((2,2), np.uint8), iterations=1)


    def update_map(self, drone: FrontierDrone):
        """
        Update the map with the occupancy grid and the confidence grid
        """

        self.map = np.where(self.occupancy_grid.get_grid() > 0, Zone.OBSTACLE, self.map)
        self.map = np.where(self.occupancy_grid.get_grid() < 0, Zone.EMPTY, self.map)
        self.map = np.where(np.logical_and(self.kill_zone.get_grid() >= 1, self.map == Zone.EMPTY), Zone.KILL_ZONE, self.map)

        if self.debug_mode:
           self.display_map(drone)

    def __setitem__(self, pos, zone):
        self.map[pos.x][pos.y] = zone

    def __getitem__(self, x, y):
        return self.map[x,y]

    
    def get_map_matrix(self):
        return self.map
    
    def add_kill_zone(self, kill_zone_pos: Vector2D):
        """
        Add a kill zone to the map
        """
        kill_zone_pos_grid = self.world_to_grid(kill_zone_pos)
        
        for x in range(kill_zone_pos_grid.x - KILL_ZONE_SIZE, kill_zone_pos_grid.x + KILL_ZONE_SIZE):
            for y in range(kill_zone_pos_grid.y - KILL_ZONE_SIZE, kill_zone_pos_grid.y + KILL_ZONE_SIZE):
                if 0 <= x < self.width and 0 <= y < self.height:
                    pos = self.grid_to_world(Vector2D(x, y))
                    self.kill_zone.add_point(pos.x, pos.y, KILL_ZONE_INCREFMENT)
                    self.kill_zone.set_point(pos.x, pos.y, min(100, self.kill_zone.get_point(pos.x, pos.y)))

    def reset_kill_zone(self):
        """
        Reset the kill zone
        """
        self.kill_zone.set_grid(np.zeros((self.width, self.height)).astype(np.float32))

    def display_map(self, drone: FrontierDrone):
        """
        Display the map
        """
        if self.identifier != 0:
            return
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.UNEXPLORED: (0, 0, 0),
            Zone.KILL_ZONE: (255, 0, 0),
            Zone.NO_COM_ZONE: (0, 255, 0)
        }

        img = np.zeros((self.width//3, self.height//3, 3), np.uint8)

        # Assign colors to each point based on the color map
        for x in range(self.width//3):
            for y in range(self.height//3):
                img[x][y] = np.array(color_map[self[Vector2D(3*x, 3*y)]])
        # plt.imsave(f"map_{self.identifier}.png", img)
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow(f"Map {drone.identifier==0}", np.transpose(img, (1, 0, 2)))
        cv2.waitKey(1)

    def world_to_grid(self, pos: Vector2D) -> Vector2D:
        """
        converts world coordinates to grid coordinates
        """
        y =  self.height - int(pos.y / self.resolution + self.height / 2)
        x = int(pos.x / self.resolution + self.width / 2)
        # crop the values to the map size
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
        return Vector2D(x, y)

    def grid_to_world(self, pos: Vector2D) -> Vector2D:
        """
        converts grid coordinates to world coordinates
        """
        y = (self.height - pos.y) * self.resolution - self.height * self.resolution / 2
        x = (pos.x - self.width / 2) * self.resolution
        return Vector2D(x, y)
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def update(self, pose, lidar, drone: FrontierDrone):
        """
        Update the map
        """
        self.update_confidence_grid(pose, lidar, drone)
        self.update_occupancy_grid(pose, lidar, drone)
        self.update_merkle(drone)
        self.update_map(drone)


    def update_merkle(self, drone: FrontierDrone):
        p1 = drone.drone_position - Vector2D(MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_LIDAR_SENSOR)
        p2 = drone.drone_position + Vector2D(MAX_RANGE_LIDAR_SENSOR, MAX_RANGE_LIDAR_SENSOR)
        p1 = self.world_to_grid(p1)
        p2 = self.world_to_grid(p2)
        self.merkle_tree.update(0,0,self.height-1, self.width-1) # TODO fix


    def merge(self, other_maps : List['Map'], drone: FrontierDrone):
        """
        Merge the map with other maps using the confidence grid : if confidence of the current map is higher than the other maps, keep the current map value, else, keep the other map value
        """        
        maps = [self] + other_maps
        new_maps = []
        while len(maps) > 1:
            for i in range(0,len(maps),2):
                if i+1<len(maps):
                    maps[i].merkle_tree.merge(maps[i+1].merkle_tree)
                new_maps.append(maps[i])
            maps = new_maps
            new_maps = []

        for other_map in other_maps:
            self.kill_zone.set_grid(np.maximum(self.kill_zone.get_grid(), other_map.kill_zone.get_grid()))

    def get_obstacle_grid(self):
        obstacle_grid = np.where(np.logical_or(self.map == Zone.OBSTACLE, np.logical_and(self.map == Zone.KILL_ZONE, self.map != Zone.NO_COM_ZONE)), 2, 0).astype(np.float64)
        return obstacle_grid
    
    def shortest_path(self, start: Vector2D, end: Vector2D, fast=False) -> Tuple[Optional[List[Vector2D]],Optional[Vector2D]]:
        """
        returns the shortest path between start and end
        Params:
            - start: start position (world coordinates)
            - end: end position (world coordinates)
        Returns:
            - path: list of positions (world coordinates)
        """
        obstacle_grid = np.where(np.logical_or(np.logical_or(self.map == Zone.OBSTACLE, self.map == Zone.UNEXPLORED), self.map == Zone.KILL_ZONE), 2, 0).astype(np.float64)

        # Save obstacle_grid for debugging
        # cv2.imwrite("obstacle_grid.png", 100*obstacle_grid)
       
        grid_start = self.world_to_grid(start)
        grid_end = self.world_to_grid(end)

        grid_path: Optional[np.ndarray] = None
        next_waypoint = None
        if fast:
            grid_path, next_waypoint  = pathfinder_fast(obstacle_grid, grid_start.array, grid_end.array)
            if next_waypoint is not None:
                next_waypoint = Vector2D(next_waypoint[0], next_waypoint[1])
                next_waypoint = self.grid_to_world(next_waypoint)
        else:
            grid_path = pathfinder(obstacle_grid, grid_start.array, grid_end.array, 6)


        if grid_path is None:
            return None,None
        path = [self.grid_to_world(Vector2D(pos[0], pos[1])) for pos in grid_path]

        return path, next_waypoint

    def update_confidence_wall_map(self):
        """
        returns the confidence wall map
        """
        self.confidence_wall_map = np.where(self.occupancy_grid.get_grid() > 0, 1, 0).astype(np.float64)
       
        self.confidence_wall_map[0:2,] = 1
        self.confidence_wall_map[-3:-1,:] = 1
        self.confidence_wall_map[:,0:2] = 1
        self.confidence_wall_map[:,-3:-1] = 1
        #erosion
        kernel = np.ones((2,2),np.uint8)
        self.confidence_wall_map = cv2.erode(self.confidence_wall_map, kernel, iterations=1)
        self.confidence_wall_map = cv2.GaussianBlur(self.confidence_wall_map, (7,7), 0)


    def get_confidence_wall_map(self, x: int, y: int):
        return self.confidence_wall_map[x, y]
