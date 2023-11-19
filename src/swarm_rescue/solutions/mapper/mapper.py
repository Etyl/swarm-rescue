from enum import Enum
import numpy as np
import cv2

from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.utils.grid import Grid
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

        self.mappers = {
            Zone.OBSTACLE: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WALL),
            #Zone.RESCUE_CENTER: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.RESCUE_CENTER),
            #Zone.WOUNDED: Mapper(area_world, resolution, lidar, DroneSemanticSensor.TypeEntity.WOUNDED_PERSON),
            Zone.EMPTY: Mapper(area_world, resolution, lidar, None)
        }

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone
        
    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]
    
    def update_grid(self, pose: Pose, semantic_lidar):
        for mapper in self.mappers.values():
            mapper.update_grid(pose, semantic_lidar)

        # TODO: optimiser: mettre à jour uniquement les points qui ont changés
        # Update map
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                for zone, mapper in self.mappers.items():
                    if mapper.binary_grid[x][y] == 1:
                        self[x, y] = zone
                        break
                else:
                    self[x, y] = Zone.INEXPLORED

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
        img = self.map_to_image()
        cv2.imshow("map", img)
        cv2.waitKey(1)

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
        # blur binary grid
        self.binary_grid = cv2.blur(self.binary_grid, (3,3))
        # # threshold binary grid
        # self.binary_grid = np.where(self.binary_grid > 0, 1, 0).astype(np.uint8)
        # # erode and dilate
        # self.binary_grid = cv2.erode(self.binary_grid, np.ones((3, 3), np.uint8), iterations=1)
