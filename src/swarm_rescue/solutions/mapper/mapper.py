from enum import Enum
import numpy as np
import cv2

from swarm_rescue.spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

    

class Map:
    def __init__(self, area_world, resolution):
        
        self.resolution = resolution
        self.x_max_grid: int = int(area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(area_world[1] / self.resolution + 0.5)
        
        self.map = np.full((self.x_max_grid, self.y_max_grid), Zone.FREE)

        self.wall_grid = Mapper(area_world, resolution, None, DroneSemanticSensor.TypeEntity.WALL)
        self.rescue_center_grid = Mapper(area_world, resolution, None, DroneSemanticSensor.TypeEntity.RESCUE_CENTER)
        self.wounded_grid = Mapper(area_world, resolution, None, DroneSemanticSensor.TypeEntity.WOUNDED)
        self.explored_grid = Mapper(area_world, resolution, None, DroneSemanticSensor.TypeEntity.WALL)


    def set_zone(self, x, y, zone : Zone):
        self.map[y][x] = zone
        
    def get_zone(self, x, y) -> Zone:
        return self.map[y][x]
    
    def update_grid(self, pose: Pose):
        self.wall_grid.update_grid(pose)
        self.rescue_center_grid.update_grid(pose)
        self.wounded_grid.update_grid(pose)
        self.explored_grid.update_grid(pose)

        # Update map
        for x in range(self.x_max_grid):
            for y in range(self.y_max_grid):
                if self.wall_grid.grid[x][y] == 1:
                    self.set_zone(x, y, Zone.OBSTACLE)
                elif self.rescue_center_grid.grid[x][y] == 1:
                    self.set_zone(x, y, Zone.RESCUE_CENTER)
                elif self.wounded_grid.grid[x][y] == 1:
                    self.set_zone(x, y, Zone.WOUNDED)
                elif self.explored_grid.grid[x][y] == 1:
                    self.set_zone(x, y, Zone.FEEE)
                else:
                    self.set_zone(x, y, Zone.INEXPLORED)

class Mapper(Grid):

    def __init__(self,
                 size_area_world,
                 resolution: float,
                 semantic_lidar,
                 zone_to_detect: DroneSemanticSensor.TypeEntity = DroneSemanticSensor.TypeEntity.WALL):
        super().__init__(size_area_world=size_area_world, resolution=resolution)

        self.size_area_world = size_area_world
        self.resolution = resolution

        self.semantic_lidar = semantic_lidar

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.zone_to_detect = zone_to_detect

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))
        self.binary_grid = np.zeros((self.x_max_grid, self.y_max_grid)).astype(np.uint8)

    def update_grid(self, pose: Pose):
        """
        Bayesian map update with new observation
        lidar : lidar data
        pose : corrected pose in world coordinates
        """
        LIDAR_DIST_CLIP = 40.0
        EMPTY_ZONE_VALUE = -0.602
        OBSTACLE_ZONE_VALUE = 2.0
        FREE_ZONE_VALUE = -4.0
        THRESHOLD_MIN = -40
        THRESHOLD_MAX = 40

        semantic_lidar_dist = np.array([data.distance for data in self.semantic_lidar if data.entity_type == self.zone_to_detect])
        semantic_lidar_angles = np.array([data.angle for data in self.semantic_lidar if data.entity_type == self.zone_to_detect])


        # Compute cos and sin of the absolute angle of the lidar
        cos_rays = np.cos(semantic_lidar_angles + pose.orientation)
        sin_rays = np.sin(semantic_lidar_angles + pose.orientation)

        max_range = MAX_RANGE_LIDAR_SENSOR * 0.9

        # For empty zones
        # points_x and point_y contains the border of detected empty zone
        # We use a value a little bit less than LIDAR_DIST_CLIP because of the noise in lidar
        lidar_dist_empty = np.maximum(semantic_lidar_dist - LIDAR_DIST_CLIP, 0.0)
        # All values of lidar_dist_empty_clip are now <= max_range
        lidar_dist_empty_clip = np.minimum(lidar_dist_empty, max_range)
        points_x = pose.position[0] + np.multiply(lidar_dist_empty_clip, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty_clip, sin_rays)

        for pt_x, pt_y in zip(points_x, points_y):
            self.add_value_along_line(pose.position[0], pose.position[1], pt_x, pt_y, EMPTY_ZONE_VALUE)

        # For obstacle zones, all values of lidar_dist are < max_range
        select_collision = lidar_dist_empty < max_range

        points_x = pose.position[0] + np.multiply(lidar_dist_empty, cos_rays)
        points_y = pose.position[1] + np.multiply(lidar_dist_empty, sin_rays)

        points_x = points_x[select_collision]
        points_y = points_y[select_collision]

        self.add_points(points_x, points_y, OBSTACLE_ZONE_VALUE)

        # the current position of the drone is free !
        self.add_points(pose.position[0], pose.position[1], FREE_ZONE_VALUE)

        # threshold values
        self.grid = np.clip(self.grid, THRESHOLD_MIN, THRESHOLD_MAX)

        # map to binary and resize to fit the world
        self.binary_grid = np.where(self.grid > 0, 1, 0)