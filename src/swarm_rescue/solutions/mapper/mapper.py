from __future__ import annotations

import numpy as np
import cv2
from typing import List, TYPE_CHECKING, Optional, Dict

from solutions.utils.types import Vector2D  # type: ignore
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR # type: ignore
from solutions.mapper.grid import Grid # type: ignore
from solutions.pathfinder.pathfinder import pathfinder, pathfinder_fast # type: ignore

from solutions.pathfinder.pathfinder import * # type: ignore
from solutions.mapper.utils import display_grid # type: ignore

if TYPE_CHECKING: # type: ignore
    from solutions.frontier_drone import FrontierDrone # type: ignore

EVERY_N = 2
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2.0
FREE_ZONE_VALUE = -8
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40
CONFIDENCE_VALUE = 500
CONFIDENCE_THRESHOLD = 1000
CONFIDENCE_THRESHOLD_MIN = 0
KILL_ZONE_SIZE = 10

class Zone:
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    UNEXPLORED = -1

class Map:
    def __init__(self, area_world: List[int], resolution: int, debug_mode: bool=False) -> None:
        self.resolution: int = resolution
        self.debug_mode: bool = debug_mode

        self.width : int = int(area_world[0] / self.resolution + 0.5)
        self.height : int = int(area_world[1] / self.resolution + 0.5)

        self.occupancy_grid : Grid = Grid(area_world, resolution)
        self.binary_occupancy_grid : np.ndarray = np.zeros((self.width, self.height)).astype(np.uint8)
        self.confidence_grid : Grid = Grid(area_world, resolution)
        self.confidence_grid_downsampled : Grid = Grid(area_world, resolution * 2)
        self.confidence_wall_map : np.ndarray = np.zeros((self.width, self.height)).astype(np.float64)
        
        self.map : np.ndarray = np.full((self.width, self.height), Zone.UNEXPLORED)

        self.rescue_center : Optional[Vector2D] = None
        self.kill_zones : Dict[int, Vector2D] = {}
        self.no_gps_zones : List[Vector2D] = []

    
    def update_confidence_grid(self, pose, lidar, drone: FrontierDrone):
        """
        Update confidence grid with drone's semantic sensor data
        """
        lidar_dist = lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = lidar.ray_angles[::EVERY_N].copy()

        # downsample_indices = np.where(lidar_dist < MAX_RANGE_LIDAR_SENSOR/2)[0]
        # downsample_indices = downsample_indices[downsample_indices % 2 == 0]

        # mask = np.zeros(lidar_dist.shape, dtype=bool)
        # mask[downsample_indices] = True
        # mask[lidar_dist > MAX_RANGE_LIDAR_SENSOR/2] = True

        downsampled_lidar_dist = lidar_dist
        downsampled_lidar_angles = lidar_angles

        world_points = np.column_stack((pose.position[0] + np.multiply(downsampled_lidar_dist, np.cos(downsampled_lidar_angles + pose.orientation)), pose.position[1] + np.multiply(downsampled_lidar_dist, np.sin(downsampled_lidar_angles + pose.orientation))))

        if not drone.gps_disabled:
            self.confidence_grid_downsampled.add_value_along_lines_confidence(pose.position[0], pose.position[1], world_points[:,0], world_points[:,1], CONFIDENCE_VALUE)
        else:
            self.confidence_grid_downsampled.add_value_along_lines_confidence(pose.position[0], pose.position[1], world_points[:,0], world_points[:,1], CONFIDENCE_VALUE/2)
        # for x, y in world_points:
        #     self.confidence_grid_downsampled.add_value_along_line_confidence(pose.position[0], pose.position[1], x, y, CONFIDENCE_VALUE)

        self.confidence_grid_downsampled.set_grid(np.clip(self.confidence_grid_downsampled.get_grid(), 0, CONFIDENCE_THRESHOLD))
        #self.confidence_grid.display(pose, title="Confidence grid of drone {}".format(drone_id))
        # Resize confidence_grid_downsampled to the size of the confidence_grid
        self.confidence_grid.set_grid(cv2.resize(self.confidence_grid_downsampled.get_grid(), (self.height, self.width), interpolation=cv2.INTER_LINEAR_EXACT))

        #display_grid(self.confidence_grid_downsampled, pose, title="Confidence grid of drone {}".format(drone_id))

    def update_occupancy_grid(self, pose, lidar, drone: FrontierDrone):
        """
        Update occupancy grid with drone's semantic sensor data
        """
         # Save values at the boundaries
        boundary_mask = np.logical_or(self.occupancy_grid.get_grid() == THRESHOLD_MIN, self.occupancy_grid.get_grid() == THRESHOLD_MAX)
        buffer = self.occupancy_grid.get_grid().copy()

        #lidar_dist = lidar.get_sensor_values()[::EVERY_N].copy()
        #lidar_angles = lidar.ray_angles[::EVERY_N].copy()

        lidar_dist = lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = lidar.ray_angles[::EVERY_N].copy()

        #downsample_indices = np.where(lidar_dist < MAX_RANGE_LIDAR_SENSOR/2)[0]
        #downsample_indices = downsample_indices[downsample_indices % 2 == 0]

        #mask = np.zeros(lidar_dist.shape, dtype=bool)
        #mask[downsample_indices] = True
        #mask[lidar_dist > MAX_RANGE_LIDAR_SENSOR/2] = True

        # downsampled_lidar_dist = lidar_dist[mask]
        # downsampled_lidar_angles = lidar_angles[mask]

        downsampled_lidar_dist = lidar_dist
        downsampled_lidar_angles = lidar_angles

        if drone.gps_disabled:
            max_range = 0.9 * MAX_RANGE_LIDAR_SENSOR
        else:
            max_range = 0.65 * MAX_RANGE_LIDAR_SENSOR
        
        lidar_dist_clip = np.minimum(np.maximum(downsampled_lidar_dist - LIDAR_DIST_CLIP, 0.0), max_range)

        world_points_free = np.column_stack((pose.position[0] + np.multiply(lidar_dist_clip, np.cos(downsampled_lidar_angles + pose.orientation)), pose.position[1] + np.multiply(lidar_dist_clip, np.sin(downsampled_lidar_angles + pose.orientation))))

        self.occupancy_grid.add_value_along_lines(pose.position[0], pose.position[1], world_points_free[:,0], world_points_free[:,1], EMPTY_ZONE_VALUE)

        lidar_dist_hit = downsampled_lidar_dist[downsampled_lidar_dist < max_range]
        lidar_angles_hit = downsampled_lidar_angles[downsampled_lidar_dist < max_range]

        world_points_hit = np.column_stack((pose.position[0] + np.multiply(lidar_dist_hit, np.cos(lidar_angles_hit + pose.orientation)), pose.position[1] + np.multiply(lidar_dist_hit, np.sin(lidar_angles_hit + pose.orientation))))

        self.occupancy_grid.set_grid(np.where(boundary_mask, buffer, self.occupancy_grid.get_grid()))
        self.occupancy_grid.add_points(world_points_hit[:,0], world_points_hit[:,1], OBSTACLE_ZONE_VALUE)
        # Mark the current position of the drone as free
        self.occupancy_grid.add_point(int(pose.position[0]), int(pose.position[1]), FREE_ZONE_VALUE)

        self.occupancy_grid.set_grid(np.clip(self.occupancy_grid.get_grid(), THRESHOLD_MIN, THRESHOLD_MAX))
        self.binary_occupancy_grid = np.where(self.occupancy_grid.get_grid() > 0, 1, -1)

        #self.filter_occupancy_grid()
        #self.occupancy_grid.display(pose, title="Occupancy grid of drone {}".format(drone_id))
        #display_grid(self.occupancy_grid, pose, title="Occupancy grid of drone {}".format(drone_id))
    
    def filter_occupancy_grid(self):
        """
        Filter the occupancy grid with a median filter
        """
        #self.binary_occupancy_grid = cv2.erode(self.binary_occupancy_grid, np.ones((2,2), np.uint8), iterations=1)


    def update_map(self, drone: FrontierDrone):
        """
        Update the map with the occupancy grid and the confidence grid
        """
        #self.map = np.where(self.confidence_grid.get_grid() > CONFIDENCE_THRESHOLD_MIN, Zone.EMPTY, self.map)
        self.map = np.where(self.occupancy_grid.get_grid() > 0, Zone.OBSTACLE, self.map)
        self.map = np.where(self.occupancy_grid.get_grid() < 0, Zone.EMPTY, self.map)
        if drone.debug_map and drone.identifier==0:
           self.display_map()

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone

    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]
    
    def get_map_matrix(self):
        return self.map
    
    def add_kill_zone(self, zone_id: int, kill_zone: Vector2D):
        """
        Add a kill zone to the map
        """
        if zone_id not in self.kill_zones:
            self.kill_zones[zone_id] = kill_zone

    def remove_kill_zone(self, zone_id: int):
        """
        Remove a kill zone from the map
        """
        if zone_id in self.kill_zones:
            self.kill_zones.pop(zone_id)

    def display_map(self):
        """
        Display the map
        """
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.UNEXPLORED: (0, 0, 0),
            1: (255, 255, 255),  # Color for points with value 1 (white)
            1000: (139, 69, 19)  # Color for points with value 1000 (brown)
        }

        img = np.zeros((self.width, self.height, 3), np.uint8)

        # Assign colors to each point based on the color map
        for x in range(self.width):
            for y in range(self.height):
                img[x][y] = np.array(color_map[self[x, y]]) #* self.confidence_grid.get_grid()[x][y] / CONFIDENCE_THRESHOLD

        # draw kill zones as rectangles
        for kill_zone in self.kill_zones.values():
            img = cv2.rectangle(img, (kill_zone.y - KILL_ZONE_SIZE, kill_zone.x - KILL_ZONE_SIZE), (kill_zone.y + KILL_ZONE_SIZE, kill_zone.x + KILL_ZONE_SIZE), (255, 0, 0), 1)

        # # Display the image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Map", np.transpose(img, (1, 0, 2)))
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
        self.update_map(drone)

    def merge(self, other_map : "Map", drone: FrontierDrone):
        """
        Merge the map with other maps using the confidence grid : if confidence of the current map is higher than the other maps, keep the current map value, else, keep the other map value
        """
        self.occupancy_grid.set_grid(np.where(other_map.confidence_grid.get_grid() > self.confidence_grid.get_grid(), other_map.occupancy_grid.get_grid(), self.occupancy_grid.get_grid()))

        self.confidence_grid.set_grid(np.maximum(self.confidence_grid.get_grid(), other_map.confidence_grid.get_grid()))

        reset = False
        for other_id in other_map.kill_zones:
            if other_id not in self.kill_zones:
                self.kill_zones[other_id] = other_map.kill_zones[other_id]
                reset = True
        if reset:
            drone.path = []
            drone.waypoint_index = None
        #self.confidence_grid.set_grid(np.maximum(self.confidence_grid.get_grid(), other_map.confidence_grid.get_grid()))
        #self.update_map()
    
    def shortest_path(self, start: Vector2D, end: Vector2D, fast=False) -> Optional[List[Vector2D]]:
        """
        returns the shortest path between start and end
        Params:
            - start: start position (world coordinates)
            - end: end position (world coordinates)
        Returns:
            - path: list of positions (world coordinates)
        """
        # self.binary_occupancy_grid = 1 if obstacle or unexplored, 0 if free
        obstacle_grid = np.where(np.logical_or(self.map == Zone.OBSTACLE, self.map == Zone.UNEXPLORED), 2, 0).astype(np.float64)
        kill_zone_grid = np.zeros((self.width, self.height)).astype(np.float64)
        # put kill zones as obstacles
        if self.kill_zones:
            for kill_zone in self.kill_zones.values():
                kill_zone = self.world_to_grid(kill_zone)
                kill_zone_grid[kill_zone.x - KILL_ZONE_SIZE:kill_zone.x + KILL_ZONE_SIZE, kill_zone.y - KILL_ZONE_SIZE:kill_zone.y + KILL_ZONE_SIZE] = 2
        #cv2.imwrite("./kill_zones.png", kill_zone_grid * 255)
        # gaussian blur tp smooth kill zones
        #kill_zone_grid = cv2.GaussianBlur(kill_zone_grid, (15, 15), 0)
        #cv2.imwrite("./kill_zones_smooth.png", kill_zone_grid * 255)

        obstacle_grid = obstacle_grid + kill_zone_grid
        # plt.imshow(obstacle_grid)
        # plt.savefig("./map.png")

        zoom_factor = 1
        #
        # if zoom_factor != 1:
        #     obstacle_grid = cv2.resize(obstacle_grid, (0, 0), fx=zoom_factor, fy=zoom_factor)
        #     # erosion
        #     # kernel = np.ones((2,2),np.uint8)
        #     # obstacle_grid = cv2.erode(obstacle_grid, kernel, iterations=2)
        # save obstacle grid as image
        #cv2.imwrite("./map.png", obstacle_grid.T * 255/2)

        grid_start = zoom_factor * self.world_to_grid(start)
        grid_end = zoom_factor * self.world_to_grid(end)

        grid_path: Optional[np.ndarray] = None
        if fast:
            grid_path  = pathfinder_fast(obstacle_grid, grid_start.array, grid_end.array)
        else:
            grid_path = pathfinder(obstacle_grid, grid_start.array, grid_end.array, 7*zoom_factor)


        if grid_path is None:
            return None
        path = [self.grid_to_world(Vector2D(pos[0] / zoom_factor, pos[1] / zoom_factor)) for pos in grid_path]

        return path

    def update_confidence_wall_map(self):
        """
        returns the confidence wall map
        """
        self.confidence_wall_map = np.where(self.occupancy_grid.get_grid() > 0, 1, 0).astype(np.float64)
        #confidence_wall_map = confidence_wall_map.astype(np.uint8)

        # Apply a gaussian blur to smooth the map
        #confidence_wall_map = cv2.GaussianBlur(confidence_wall_map, (3, 3), 0)
        # confidence_wall_map = confidence_wall_map.astype(np.float64)
        # confidence_wall_map = confidence_wall_map * self.confidence_grid.get_grid() / CONFIDENCE_THRESHOLD
        #confidence_wall_map = confidence_wall_map * self.confidence_grid.get_grid() / CONFIDENCE_THRESHOLD
        self.confidence_wall_map[0:2,] = 1
        self.confidence_wall_map[-3:-1,:] = 1
        self.confidence_wall_map[:,0:2] = 1
        self.confidence_wall_map[:,-3:-1] = 1
        #erosion
        kernel = np.ones((2,2),np.uint8)
        self.confidence_wall_map = cv2.erode(self.confidence_wall_map, kernel, iterations=1)
        # get the botder of the binary grid with Cannys algorithm
        #self.confidence_map = cv2.Canny(self.confidence_map, 0, 1)
        # apply a gaussian blur to the confidence map
        self.confidence_wall_map = cv2.GaussianBlur(self.confidence_wall_map, (7,7), 0)


    def get_confidence_wall_map(self, x: int, y: int):
        return self.confidence_wall_map[x, y]
