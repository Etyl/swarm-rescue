import math
import numpy as np
import cv2

from spg_overlay.utils.pose import Pose

class Grid:
    """Simple grid"""

    def __init__(self,
                 size_area_world : tuple,
                 resolution: float):
        self.size_area_world = size_area_world
        self.resolution = resolution

        self.x_max_grid: int = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid: int = int(self.size_area_world[1] / self.resolution + 0.5)

        self.grid = np.zeros((self.x_max_grid, self.y_max_grid))

    def _conv_world_to_grid_v(self, x_world : np.ndarray, y_world : np.ndarray):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : list of x and y coordinates in m
        """

        x_grid = (x_world + self.size_area_world[0] / 2) / self.resolution
        y_grid = (-y_world + self.size_area_world[1] / 2) / self.resolution

        x_grid = x_grid.astype(int)
        y_grid = y_grid.astype(int)

        return x_grid, y_grid
    
    def _conv_world_to_grid(self, x_world : float, y_world : float):
        """
        Convert from world coordinates to map coordinates (i.e. cell index in the grid map)
        x_world, y_world : float of x and y coordinates in m
        """

        x_grid = (x_world + self.size_area_world[0] / 2) / self.resolution
        y_grid = (-y_world + self.size_area_world[1] / 2) / self.resolution

        x_grid = int(x_grid)
        y_grid = int(y_grid)


        return x_grid, y_grid

    def _conv_grid_to_world_v(self, x_grid : np.ndarray, y_grid : np.ndarray):
        """
        Convert from map coordinates to world coordinates
        x_grid, y_grid : list of x and y coordinates in cell numbers (~pixels)
        """
 
        x_grid = x_grid.astype(float)
        y_grid = y_grid.astype(float)

        x_world = -self.size_area_world[0] / 2 + x_grid * self.resolution
        y_world = self.size_area_world[1] / 2 - y_grid * self.resolution

        x_world = x_world.astype(float)
        y_world = y_world.astype(float)

        return x_world, y_world
    
    def _conv_grid_to_world(self, x_grid : int, y_grid : int):
        """
        Convert from map coordinates to world coordinates
        x_grid, y_grid : int of x and y coordinates in cell numbers (~pixels)
        """
 
        x_world = -self.size_area_world[0] / 2 + x_grid * self.resolution
        y_world = self.size_area_world[1] / 2 - y_grid * self.resolution

        return x_world, y_world

    def add_value_along_line(self, x_0: float, y_0: float, x_1: float, y_1: float, val):
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

        # add value to the points
        self.grid[points[0], points[1]] += val

    def add_points(self, points_x : np.ndarray, points_y : np.ndarray, val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y :  list of x and y coordinates in m
        val :  value to add to the cells of the points
        """

        x_px, y_px = self._conv_world_to_grid_v(points_x, points_y)

        select = np.logical_and(np.logical_and(x_px >= 0, x_px < self.x_max_grid),
                                np.logical_and(y_px >= 0, y_px < self.y_max_grid))
        x_px = x_px[select]
        y_px = y_px[select]
        self.grid[x_px, y_px] += val

    def add_point(self, x : float, y : float, val):
        """
        Add a value to a point, input coordinates in meters
        x, y :  x and y coordinates in m
        val :  value to add to the cell of the point
        """
        x_px, y_px = self._conv_world_to_grid(x, y)

        if 0 <= x_px < self.x_max_grid and 0 <= y_px < self.y_max_grid:
            self.grid[x_px, y_px] += val

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
            dist_from_start = (np.abs((x - x_0)) + np.abs((y - y_0))) * self.resolution / 15

            added_value = val / max(1, dist_from_start)  # Variation en fonction de la distance

            self.grid[x, y] += added_value

    def get_grid(self):
        return self.grid
    
    def set_grid(self, grid):
        self.grid = grid

    def display(self, robot_pose: Pose, title="grid"):
        """
        Screen display of grid and robot pose,
        using opencv (faster than the matplotlib version)
        robot_pose : [x, y, theta] nparray, corrected robot pose
        """
        img = self.grid.T
        img = img - img.min()
        img = img / img.max() * 255
        img = np.uint8(img)
        img_color = cv2.applyColorMap(src=img, colormap=cv2.COLORMAP_JET)

        # resize image
        img_color = cv2.resize(img_color, (0, 0), fx=self.resolution//2, fy=self.resolution//2, interpolation=cv2.INTER_NEAREST)

        pt2_x = robot_pose.position[0] + np.cos(robot_pose.orientation) * 20
        pt2_y = robot_pose.position[1] + np.sin(robot_pose.orientation) * 20
        pt2_x, pt2_y = self._conv_world_to_grid(pt2_x, pt2_y)

        pt1_x, pt1_y = self._conv_world_to_grid(robot_pose.position[0], robot_pose.position[1])

        pt1 = (int(pt1_x * self.resolution//2), int(pt1_y * self.resolution//2))
        pt2 = (int(pt2_x * self.resolution//2), int(pt2_y * self.resolution//2))
        cv2.arrowedLine(img=img_color, pt1=pt1, pt2=pt2,
                        color=(0, 0, 255), thickness=2)
        cv2.imshow(title, img_color)
        cv2.waitKey(1)
