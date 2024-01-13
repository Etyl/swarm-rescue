from enum import Enum
from solutions.pathfinder.pathfinder import *
import numpy as np
import math
import cv2
from spg.agent.sensor.sensor import Sensor

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

from solutions.mapper.utils import Grid, world_to_grid, grid_to_world


EVERY_N = 1
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2
FREE_ZONE_VALUE = -8
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40
CONFIDENCE_VALUE = 20
CONFIDENCE_THRESHOLD = 100
CONFIDENCE_THRESHOLD_MIN = 0

class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class Map():
    def __init__(self, area_world, resolution, debug_mode=False):
        self.resolution = resolution
        self.debug_mode = debug_mode

        self.area_world = area_world

        self.width = int(area_world[0] / self.resolution + 0.5)
        self.height = int(area_world[1] / self.resolution + 0.5)

        self.occupancy_grid = Grid(area_world, resolution)
        self.binary_occupancy_grid = np.zeros((self.width, self.height)).astype(np.uint8)
        self.confidence_grid = Grid(area_world, resolution)
        
        self.map = np.full((self.width, self.height), Zone.INEXPLORED)

        self.rescue_center = None
        self.kill_zones = []
        self.no_gps_zones = []
        self.wounded_persons = []
    
    def update_confidence_grid(self, pose, drone_lidar: Sensor):
        """
        Update confidence grid with drone's semantic sensor data
        """
        lidar_dist = drone_lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = drone_lidar.ray_angles[::EVERY_N].copy()

        world_points = np.array([(pose.position[0] + d*math.cos(a + pose.orientation), pose.position[1] + d*math.sin(a + pose.orientation)) for d, a in zip(lidar_dist, lidar_angles)])

        for x, y in world_points:
            self.confidence_grid.add_value_along_line_confidence(pose.position[0], pose.position[1], x, y, CONFIDENCE_VALUE)

        self.confidence_grid.grid = np.clip(self.confidence_grid.grid, 0, CONFIDENCE_THRESHOLD)

        #self.confidence_grid.display(pose)

    def update_occupancy_grid(self, pose, drone_lidar: Sensor):
        """
        Update occupancy grid with drone's semantic sensor data
        """
         # Save values at the boundaries
        boundary_mask = np.logical_or(self.occupancy_grid.grid == THRESHOLD_MIN, self.occupancy_grid.grid == THRESHOLD_MAX)
        buffer = self.occupancy_grid.grid.copy()

        lidar_dist = drone_lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = drone_lidar.ray_angles[::EVERY_N].copy()

        max_range = 0.9 * MAX_RANGE_LIDAR_SENSOR
        
        lidar_dist_clip = np.minimum(np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0), max_range)

        world_points_free = np.array([(pose.position[0] + d*math.cos(a + pose.orientation), pose.position[1] + d*math.sin(a + pose.orientation)) for d, a in zip(lidar_dist_clip, lidar_angles)])

        for x, y in world_points_free:
            self.occupancy_grid.add_value_along_line(pose.position[0], pose.position[1], x, y, EMPTY_ZONE_VALUE)

        lidar_dist_hit = lidar_dist[lidar_dist < max_range]
        lidar_angles_hit = lidar_angles[lidar_dist < max_range]

        world_points_hit = np.array([(pose.position[0] + d*math.cos(a + pose.orientation), pose.position[1] + d*math.sin(a + pose.orientation)) for d, a in zip(lidar_dist_hit, lidar_angles_hit)])

        self.occupancy_grid.grid = np.where(boundary_mask, buffer, self.occupancy_grid.grid)
        self.occupancy_grid.add_points(world_points_hit[:,0], world_points_hit[:,1], OBSTACLE_ZONE_VALUE)
        # Mark the current position of the drone as free
        #self.occupancy_grid.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        self.occupancy_grid.grid = np.clip(self.occupancy_grid.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.binary_occupancy_grid = np.where(self.occupancy_grid.grid > 0, 1, -1)

        #self.filter_occupancy_grid()

        #self.occupancy_grid.display(pose)
    
    def filter_occupancy_grid(self):
        """
        Filter the occupancy grid with a median filter
        """
        #self.binary_occupancy_grid = cv2.erode(self.binary_occupancy_grid, np.ones((2,2), np.uint8), iterations=1)


    def update_map(self):
        """
        Update the map with the occupancy grid and the confidence grid
        """
        #self.map = np.where(self.confidence_grid.grid > CONFIDENCE_THRESHOLD_MIN, Zone.EMPTY, self.map)
        self.map = np.where(self.occupancy_grid.grid > 0, Zone.OBSTACLE, self.map)
        self.map = np.where(self.occupancy_grid.grid < 0, Zone.EMPTY, self.map)

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone

    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]
    
    def get_map_matrix(self):
        return self.map

    def display_map(self):
        """
        Display the map
        """
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.INEXPLORED: (0, 0, 0),
            1: (255, 255, 255),  # Color for points with value 1 (white)
            1000: (139, 69, 19)  # Color for points with value 1000 (brown)
        }

        img = np.zeros((self.width, self.height, 3), np.uint8)

        # Assign colors to each point based on the color map
        for x in range(self.width):
            for y in range(self.height):
                img[x][y] = color_map[self[x, y]]


        # # Display the image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        cv2.imshow("Map", np.transpose(img, (1, 0, 2)))
        cv2.waitKey(1)

    def world_to_grid(self, pos: Pose):
        """
        converts world coordinates to grid coordinates
        """
        y =  self.height - int(pos[1] / self.resolution + self.height / 2)
        x = int(pos[0] / self.resolution + self.width / 2)
        return np.array([x, y])

    def grid_to_world(self, pos):
        """
        converts grid coordinates to world coordinates
        """
        y = (self.height - pos[1]) * self.resolution - self.height * self.resolution / 2
        x = (pos[0] - self.width / 2) * self.resolution
        return [x, y]
    
    def get_width(self):
        return self.width
    
    def get_height(self):
        return self.height
    
    def update(self, pose, drone_lidar: Sensor):
        """
        Update the map
        """
        self.update_confidence_grid(pose, drone_lidar)
        self.update_occupancy_grid(pose, drone_lidar)
        self.update_map()
    
    def shortest_path(self, start: Pose, end: Pose):
        """
        returns the shortest path between start and end
        Params:
            - start: start position (world coordinates)
            - end: end position (world coordinates)
        Returns:
            - path: list of positions (world coordinates)
        """
        obstacle_grid = (self.binary_occupancy_grid == -1).astype(np.uint8)

        # plt.imshow(obstacle_grid)
        # plt.savefig("./map.png")

        zoom_factor = 3

        if zoom_factor != 1:
            obstacle_grid = cv2.resize(obstacle_grid, (0, 0), fx=zoom_factor, fy=zoom_factor)
            # erosion
            kernel = np.ones((2,2),np.uint8)
            obstacle_grid = cv2.erode(obstacle_grid, kernel, iterations=2)

        adjusted_start = [start[0], start[1]]
        grid_start = [coord * zoom_factor for coord in self.world_to_grid(adjusted_start)]
        grid_end = [coord * zoom_factor for coord in self.world_to_grid(end)]

        grid_path = pathfinder(obstacle_grid, grid_start, grid_end)

        path = [self.grid_to_world([pos[0] / zoom_factor, pos[1] / zoom_factor]) for pos in grid_path]

        path.reverse()
        return path

