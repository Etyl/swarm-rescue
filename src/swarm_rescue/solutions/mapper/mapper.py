from enum import Enum
import numpy as np
import cv2
import time

from spg_overlay.utils.pose import Pose
from spg_overlay.utils.constants import MAX_RANGE_LIDAR_SENSOR
#from solutions.mapper.grid_old import Grid
from solutions.mapper.grid import Grid

from solutions.pathfinder.pathfinder import *
from solutions.mapper.utils import display_grid

EVERY_N = 2
LIDAR_DIST_CLIP = 40.0
EMPTY_ZONE_VALUE = -1
OBSTACLE_ZONE_VALUE = 2
FREE_ZONE_VALUE = -8
THRESHOLD_MIN = -40
THRESHOLD_MAX = 40
CONFIDENCE_VALUE = 500
CONFIDENCE_THRESHOLD = 1000
CONFIDENCE_THRESHOLD_MIN = 0
KILL_ZONE_SIZE = 5

class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class Map():
    def __init__(self, drone, area_world, drone_lidar, resolution, identifier, debug_mode=False):
        self.resolution = resolution
        self.debug_mode = debug_mode
        self.drone_lidar = drone_lidar
        self.drone_id = identifier
        self.drone = drone

        self.area_world = area_world

        self.width = int(area_world[0] / self.resolution + 0.5)
        self.height = int(area_world[1] / self.resolution + 0.5)

        self.occupancy_grid = Grid(area_world, resolution)
        self.binary_occupancy_grid = np.zeros((self.width, self.height)).astype(np.uint8)
        self.confidence_grid = Grid(area_world, resolution)
        self.confidence_grid_downsampled = Grid(area_world, resolution * 4)
        
        self.map = np.full((self.width, self.height), Zone.INEXPLORED)

        self.rescue_center = None
        self.kill_zones = {}
        self.no_gps_zones = []
        self.wounded_persons = []
    
    def update_confidence_grid(self, pose):
        """
        Update confidence grid with drone's semantic sensor data
        """
        lidar_dist = self.drone_lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.drone_lidar.ray_angles[::EVERY_N].copy()

        downsample_indices = np.where(lidar_dist < MAX_RANGE_LIDAR_SENSOR/2)[0]
        downsample_indices = downsample_indices[downsample_indices % 2 == 0]

        mask = np.zeros(lidar_dist.shape, dtype=bool)
        mask[downsample_indices] = True
        mask[lidar_dist > MAX_RANGE_LIDAR_SENSOR/2] = True

        downsampled_lidar_dist = lidar_dist[mask]
        downsampled_lidar_angles = lidar_angles[mask]

        world_points = np.column_stack((pose.position[0] + np.multiply(downsampled_lidar_dist, np.cos(downsampled_lidar_angles + pose.orientation)), pose.position[1] + np.multiply(downsampled_lidar_dist, np.sin(downsampled_lidar_angles + pose.orientation))))

        self.confidence_grid_downsampled.add_value_along_lines_confidence(pose.position[0], pose.position[1], world_points[:,0], world_points[:,1], CONFIDENCE_VALUE)
        # for x, y in world_points:
        #     self.confidence_grid_downsampled.add_value_along_line_confidence(pose.position[0], pose.position[1], x, y, CONFIDENCE_VALUE)

        self.confidence_grid_downsampled.set_grid(np.clip(self.confidence_grid_downsampled.get_grid(), 0, CONFIDENCE_THRESHOLD))
        #self.confidence_grid.display(pose, title="Confidence grid of drone {}".format(self.drone_id))
        # Resize confidence_grid_downsampled to the size of the confidence_grid
        self.confidence_grid.set_grid(cv2.resize(self.confidence_grid_downsampled.get_grid(), (self.height, self.width), interpolation=cv2.INTER_LINEAR_EXACT))

        #display_grid(self.confidence_grid_downsampled, pose, title="Confidence grid of drone {}".format(self.drone_id))

    def update_occupancy_grid(self, pose):
        """
        Update occupancy grid with drone's semantic sensor data
        """
         # Save values at the boundaries
        boundary_mask = np.logical_or(self.occupancy_grid.get_grid() == THRESHOLD_MIN, self.occupancy_grid.get_grid() == THRESHOLD_MAX)
        buffer = self.occupancy_grid.get_grid().copy()

        lidar_dist = self.drone_lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.drone_lidar.ray_angles[::EVERY_N].copy()

        lidar_dist = self.drone_lidar.get_sensor_values()[::EVERY_N].copy()
        lidar_angles = self.drone_lidar.ray_angles[::EVERY_N].copy()

        downsample_indices = np.where(lidar_dist < MAX_RANGE_LIDAR_SENSOR/2)[0]
        downsample_indices = downsample_indices[downsample_indices % 2 == 0]

        mask = np.zeros(lidar_dist.shape, dtype=bool)
        mask[downsample_indices] = True
        mask[lidar_dist > MAX_RANGE_LIDAR_SENSOR/2] = True

        downsampled_lidar_dist = lidar_dist[mask]
        downsampled_lidar_angles = lidar_angles[mask]

        max_range = 0.9 * MAX_RANGE_LIDAR_SENSOR
        
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
        #self.occupancy_grid.display(pose, title="Occupancy grid of drone {}".format(self.drone_id))
        #display_grid(self.occupancy_grid, pose, title="Occupancy grid of drone {}".format(self.drone_id))
    
    def filter_occupancy_grid(self):
        """
        Filter the occupancy grid with a median filter
        """
        #self.binary_occupancy_grid = cv2.erode(self.binary_occupancy_grid, np.ones((2,2), np.uint8), iterations=1)


    def update_map(self):
        """
        Update the map with the occupancy grid and the confidence grid
        """
        #self.map = np.where(self.confidence_grid.get_grid() > CONFIDENCE_THRESHOLD_MIN, Zone.EMPTY, self.map)
        self.map = np.where(self.occupancy_grid.get_grid() > 0, Zone.OBSTACLE, self.map)
        self.map = np.where(self.occupancy_grid.get_grid() < 0, Zone.EMPTY, self.map)
        #self.display_map()

    def __setitem__(self, pos, zone):
        x,y = pos
        self.map[x][y] = zone

    def __getitem__(self, pos):
        x,y = pos
        return self.map[x][y]
    
    def get_map_matrix(self):
        return self.map
    
    def add_kill_zone(self, id, kill_zone):
        """
        Add a kill zone to the map
        """
        kill_zone = self.world_to_grid(kill_zone)
        if id not in self.kill_zones:
            self.kill_zones[id] = kill_zone

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
                img[x][y] = np.array(color_map[self[x, y]]) * self.confidence_grid.get_grid()[x][y] / CONFIDENCE_THRESHOLD

        # draw kill zones as rectangles
        for kill_zone in self.kill_zones:
            img = cv2.rectangle(img, (kill_zone[1] - KILL_ZONE_SIZE, kill_zone[0] - KILL_ZONE_SIZE), (kill_zone[1] + KILL_ZONE_SIZE, kill_zone[0] + KILL_ZONE_SIZE), (255, 0, 0), 1)

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
        # crop the values to the map size
        x = max(0, min(x, self.width - 1))
        y = max(0, min(y, self.height - 1))
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
    
    def update(self, pose):
        """
        Update the map
        """
        self.update_confidence_grid(pose)
        self.update_occupancy_grid(pose)
        self.update_map()

    def merge(self, other_map : "Map"):
        """
        Merge the map with other maps using the confidence grid : if confidence of the current map is higher than the other maps, keep the current map value, else, keep the other map value
        """
        self.occupancy_grid.set_grid(np.where(self.confidence_grid.get_grid() > other_map.confidence_grid.get_grid(), self.occupancy_grid.get_grid(), other_map.occupancy_grid.get_grid()))
        reset = False
        for other_id in other_map.kill_zones:
            if other_id not in self.kill_zones:
                self.kill_zones[other_id] = other_map.kill_zones[other_id]
                reset = True
        if reset:
            self.drone.path = []
            self.drone.nextWaypoint = None
        #self.confidence_grid.set_grid(np.maximum(self.confidence_grid.get_grid(), other_map.confidence_grid.get_grid()))
        #self.update_map()
    
    def shortest_path(self, start: Pose, end: Pose):
        """
        returns the shortest path between start and end
        Params:
            - start: start position (world coordinates)
            - end: end position (world coordinates)
        Returns:
            - path: list of positions (world coordinates)
        """
        # self.binary_occupancy_grid = 1 if obstacle or unexplored, 0 if free
        obstacle_grid = np.where(np.logical_or(self.map == Zone.OBSTACLE, self.map == Zone.INEXPLORED), 2, 0).astype(np.float64)
        kill_zone_grid = np.zeros((self.width, self.height)).astype(np.float64)
        # put kill zones as obstacles
        for kill_zone in self.kill_zones.values():
            kill_zone_grid[kill_zone[0] - KILL_ZONE_SIZE:kill_zone[0] + KILL_ZONE_SIZE, kill_zone[1] - KILL_ZONE_SIZE:kill_zone[1] + KILL_ZONE_SIZE] = 2
        #cv2.imwrite("./kill_zones.png", kill_zone_grid * 255)
        # gaussian blur tp smooth kill zones
        #kill_zone_grid = cv2.GaussianBlur(kill_zone_grid, (15, 15), 0)
        #cv2.imwrite("./kill_zones_smooth.png", kill_zone_grid * 255)

        obstacle_grid = obstacle_grid + kill_zone_grid
        # plt.imshow(obstacle_grid)
        # plt.savefig("./map.png")

        zoom_factor = 3

        if zoom_factor != 1:
            obstacle_grid = cv2.resize(obstacle_grid, (0, 0), fx=zoom_factor, fy=zoom_factor)
            # erosion
            # kernel = np.ones((2,2),np.uint8)
            # obstacle_grid = cv2.erode(obstacle_grid, kernel, iterations=2)
        # save obstacle grid as image
        #cv2.imwrite("./map.png", obstacle_grid.T * 255/2)
        adjusted_start = [start[0], start[1]]
        grid_start = [coord * zoom_factor for coord in self.world_to_grid(adjusted_start)]
        grid_end = [coord * zoom_factor for coord in self.world_to_grid(end)]

        grid_path = pathfinder(obstacle_grid, grid_start, grid_end, robot_radius=40//zoom_factor)

        if grid_path is None:
            return None
        path = [self.grid_to_world([pos[0] / zoom_factor, pos[1] / zoom_factor]) for pos in grid_path]

        path.reverse()
        return path
