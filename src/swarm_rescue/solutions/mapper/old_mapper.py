from enum import Enum
import numpy as np
import math
import cv2

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR

from solutions.pathfinder.pathfinder import *
from solutions.mapper.utils import Grid

EVERY_N = 1
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -0.602
OBSTACLE_ZONE_VALUE = 2.0
FREE_ZONE_VALUE = -8
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40
CONFIDENCE_VALUE = 10
CONFIDENCE_THRESHOLD = 500


class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class MapOld:
    def __init__(self, area_world, resolution, lidar, debug_mode=False):

        self.resolution = resolution
        self.debug_mode = debug_mode
        self.x_max_grid: int = int(area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(area_world[1] / self.resolution + 0.5)

        self.map = np.full((self.x_max_grid, self.y_max_grid), Zone.INEXPLORED)

        self.mappers = {
            Zone.OBSTACLE: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WALL)
            #Zone.RESCUE_CENTER: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.RESCUE_CENTER),
            #Zone.WOUNDED: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WOUNDED_PERSON),
        }

        self.confidence_wall_map = np.zeros((self.x_max_grid, self.y_max_grid))
        self.confidence_map = np.zeros((self.x_max_grid, self.y_max_grid))

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone

    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]

    def update(self, pose: Pose, semantic_lidar):

        for zone, mapper in self.mappers.items():
            mapper.update_grid(pose, semantic_lidar)
            self.confidence_wall_map = mapper.get_confidence_wall_map()
            self.confidence_map = mapper.get_confidence_map()

            for x, y in zip(*mapper.changed_points):
                if mapper.binary_grid[x][y] == 1:
                    self[x, y] = zone
                elif mapper.binary_grid[x][y] == -1:
                    self[x, y] = Zone.EMPTY
                else:
                    self[x, y] = Zone.INEXPLORED

        if self.debug_mode:
            self.display_map()

    def map_to_image(self):
        """
        returns the map as an image
        """
        color_map = {
            Zone.OBSTACLE: (50, 100, 200),
            Zone.EMPTY: (255, 255, 255),
            Zone.WOUNDED: (0, 0, 255),
            Zone.RESCUE_CENTER: (255, 255, 0),
            Zone.INEXPLORED: (0, 0, 0)
        }

        img = np.zeros((self.x_max_grid, self.y_max_grid, 3), np.uint8)
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                img[x][y] = color_map[self[x, y]]
        # zoom image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)
        return np.transpose(img, (1, 0, 2))

    def display_map(self):
        """
        displays the map
        """
        x_max_grid, y_max_grid = self.map.shape

        #Define color map
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
                img[x][y] = color_map[self[x, y]]


        # Convert coordinates to integers and assign blue color to the path
        # for coord in path:
        #     x, y = map(int, coord)
        #     img[x, y] = (0, 0, 255)

        # Zoom image
        img = cv2.resize(img, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST)

        # Display the image
        cv2.imshow("mapper_debug" + str(id), np.transpose(img, (1, 0, 2)))
        cv2.waitKey(1)


        # img = cv2.resize(self.confidence_wall_map, (0, 0), fx=5, fy=5, interpolation=cv2.INTER_NEAREST) / CONFIDENCE_THRESHOLD * 255
        # cv2.imshow("map", img.T)
        # cv2.waitKey(1)

    def world_to_grid(self, pos: Pose):
        """
        converts world coordinates to grid coordinates
        """
        y =  self.y_max_grid - int(pos[1] / self.resolution + self.y_max_grid / 2)
        x = int(pos[0] / self.resolution + self.x_max_grid / 2)
        return np.array([x, y])

    def grid_to_world(self, pos):
        """
        converts grid coordinates to world coordinates
        """
        y = (self.y_max_grid - pos[1]) * self.resolution - self.y_max_grid * self.resolution / 2
        x = (pos[0] - self.x_max_grid / 2) * self.resolution
        return [x, y]

    def shortest_path(self, start: Pose, end: Pose):
        """
        returns the shortest path between start and end
        Params:
            - start: start position (world coordinates)
            - end: end position (world coordinates)
        Returns:
            - path: list of positions (world coordinates)
        """
        obstacle_grid = (self.mappers[Zone.OBSTACLE].binary_grid == -1).astype(np.uint8)

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

    def merge_maps(self, maps):
        """
        merges the maps
        """
        for zone, _ in self.mappers.items():
            for map in maps:
                mask = map.confidence_map > self.confidence_map
                self.mappers[zone].grid = np.where(mask, map.mappers[zone].grid, self.mappers[zone].grid)
        for map in maps:
            mask = map.confidence_map > self.confidence_map
            # save mask as an image
            img = np.where(mask, 255, 0)
            cv2.imwrite("./mask.png", img)
            self.confidence_map = np.clip(self.confidence_map + map.confidence_map, 0, CONFIDENCE_THRESHOLD)
            self.confidence_wall_map = np.where(mask, map.confidence_wall_map, self.confidence_wall_map)



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

        self.previous_binary_grid = np.zeros((self.x_max_grid, self.y_max_grid)).astype(np.uint8)
        self.binary_grid = np.zeros((self.x_max_grid, self.y_max_grid)).astype(np.uint8)

        self.changed_points = np.array([])

        self.confidence_wall_map = np.zeros((self.x_max_grid, self.y_max_grid))
        self.confidence_map = np.zeros((self.x_max_grid, self.y_max_grid))

    def update_grid(self, pose: Pose, semantic_lidar):
        """
        Updates the Bayesian map with new observations from LIDAR and semantic LIDAR.

        Args:
        - pose: Corrected pose in world coordinates
        - semantic_lidar: Semantic LIDAR data
        """
        self.confidence_wall_map = np.zeros((self.x_max_grid, self.y_max_grid))
        # Extract data from LIDAR
        self.lock_grid = np.logical_or(self.grid == THRESHOLD_MIN, self.grid == THRESHOLD_MAX)

        self.buffer = self.grid.copy()

        lidar_dist = self.lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.lidar.ray_angles[::EVERY_N].copy()

        # Extract relevant angles from semantic LIDAR for wounded persons
        semantic_lidar_angles = [data.angle for data in semantic_lidar if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON]

        # Filter LIDAR data based on semantic LIDAR angles
        if semantic_lidar_angles:
            min_angle = min(semantic_lidar_angles)
            max_angle = max(semantic_lidar_angles)
            lidar_dist = np.array([lidar_dist[i] for i in range(len(lidar_dist)) if lidar_angles[i] < min_angle - 0.1 or lidar_angles[i] > max_angle + 0.1])
            lidar_angles = np.array([lidar_angles[i] for i in range(len(lidar_angles)) if lidar_angles[i] < min_angle - 0.1 or lidar_angles[i] > max_angle + 0.1])

        # Compute cos and sin of lidar angles
        cos_rays = np.cos(lidar_angles + pose.orientation)
        sin_rays = np.sin(lidar_angles + pose.orientation)

        # Define max range and process empty zones
        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9
        lidar_dist_empty = np.maximum(lidar_dist - LIDAR_DIST_CLIP, 0.0)
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)
            self.add_value_along_line_confidence(pose.position[0], pose.position[1], pt_x, pt_y, CONFIDENCE_VALUE)

        # Process obstacle zones
        select_collision = lidar_dist < max_range
        points_x = pose.position[0] + np.multiply(lidar_dist, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist, sin_rays)
        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # Mark the current position of the drone as free
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # Apply threshold values and update the grid
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)
        self.grid[self.lock_grid] = self.buffer[self.lock_grid]

        # Convert grid values into binary representations
        self.binary_grid = np.where(self.grid > 0, 1, self.grid)
        self.binary_grid = np.where(self.grid < 0, -1, self.binary_grid)
        self.changed_points = np.where(self.binary_grid != self.previous_binary_grid)
        self.previous_binary_grid = self.binary_grid.copy()

        # Update confidence map all the values of binary grid that are not 0
        self.confidence_wall_map = np.where(self.binary_grid > 0, 1, self.confidence_wall_map)
        #erosion
        kernel = np.ones((2,2),np.uint8)
        self.confidence_wall_map = cv2.erode(self.confidence_wall_map, kernel, iterations=1)
        # get the botder of the binary grid with Cannys algorithm
        #self.confidence_wall_map = cv2.Canny(self.confidence_wall_map, 0, 1)
        # apply a gaussian blur to the confidence map
        self.confidence_wall_map = cv2.GaussianBlur(self.confidence_wall_map, (7,7), 0)

        self.confidence_map = np.clip(self.confidence_map, 0, CONFIDENCE_THRESHOLD)


    def get_confidence_wall_map(self):
        """
        Returns the confidence wall map of the map
        """
        return self.confidence_wall_map

    def get_confidence_map(self):
        """
        Returns the confidence map of the map
        """
        return self.confidence_map

    def add_value_along_line_confidence(self, x_0: float, y_0: float, x_1: float, y_1: float, val):
        """
        Add a value to a line of points using Bresenham algorithm, input in world coordinates
        x_0, y_0 : starting point coordinates in m
        x_1, y_1 : end point coordinates in m
        val : value to add to each cell of the line
        """

        if math.isnan(x_0) or math.isnan(y_0) or math.isnan(x_1) or math.isnan(y_1):
            return

        # convert to pixels
        x_start, y_start = self._conv_world_to_grid(x_0, y_0)
        x_end, y_end = self._conv_world_to_grid(x_1, y_1)

        if x_start < 0 or x_start >= self.x_max_grid or y_start < 0 or y_start >= self.y_max_grid:
            # print("add_value_along_line: warning ray exits 1")
            return

        if x_end < 0 or x_end >= self.x_max_grid or y_end < 0 or y_end >= self.y_max_grid:
            # print("add_value_along_line: warning ray exits 2")
            return

        # Bresenham line drawing
        dx = x_end - x_start
        dy = y_end - y_start
        is_steep = abs(dy) > abs(dx)  # determine how steep the line is
        if is_steep:  # rotate line
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end
        # swap start and end points if necessary and store swap state
        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start
        dx = x_end - x_start  # recalculate differentials
        dy = y_end - y_start  # recalculate differentials
        error = int(dx / 2.0)  # calculate error
        y_step = 1 if y_start < y_end else -1
        # iterate over bounding box generating points between start and end
        y = y_start
        points = []
        for x in range(x_start, x_end + 1):
            coord = [y, x] if is_steep else [x, y]
            points.append(coord)
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
        points = np.array(points).T

        for i in range(len(points[0])):
            x, y = points[0][i], points[1][i]
            dist_from_start = np.abs((x - x_0)) + np.abs((y - y_0))

            added_value = val / max(1, dist_from_start)  # Variation en fonction de la distance

            self.confidence_map[x, y] += added_value
