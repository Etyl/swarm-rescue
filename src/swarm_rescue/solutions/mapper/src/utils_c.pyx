import numpy as np
cimport numpy as cnp
from typing import List, Tuple
from libc.math cimport sqrt, isnan
from libc.stdlib cimport malloc, free

cnp.import_array()  # Necessary to call when using NumPy with Cython

DTYPE = np.float64
ctypedef cnp.float64_t DTYPE_t
cimport cython

cdef class Grid:
    cdef public int x_max_grid, y_max_grid
    cdef public float resolution
    cdef public tuple size_area_world
    cdef public object grid  # Using a generic object type for the grid

    def __cinit__(self, tuple size_area_world, float resolution):
        self.size_area_world = size_area_world
        self.resolution = resolution
        self.x_max_grid = int(self.size_area_world[0] / self.resolution + 0.5)
        self.y_max_grid = int(self.size_area_world[1] / self.resolution + 0.5)
        self.grid = np.zeros((self.x_max_grid, self.y_max_grid), dtype=DTYPE)

    cdef tuple conv_world_to_grid(self, float x_world, float y_world):
        cdef int x_grid, y_grid
        x_grid = int((x_world + self.size_area_world[0] / 2) / self.resolution)
        y_grid = int((-y_world + self.size_area_world[1] / 2) / self.resolution)
        return x_grid, y_grid

    cdef tuple conv_grid_to_world(self, int x_grid, int y_grid):
        cdef float x_world, y_world
        x_world = (x_grid * self.resolution - self.size_area_world[0] / 2)
        y_world = (-y_grid * self.resolution + self.size_area_world[1] / 2)
        return x_world, y_world

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef add_value_along_line(self, float x_0, float y_0, float x_1, float y_1, DTYPE_t val):
        cdef int x_start, y_start, x_end, y_end, dx, dy, is_steep, y_step, y, x
        cdef float error
        cdef DTYPE_t[:, :] grid_view  # Declare memoryview for direct array access

        # Handle NaNs
        if isnan(x_0) or isnan(y_0) or isnan(x_1) or isnan(y_1):
            return

        # Convert to grid coordinates
        x_start, y_start = self.conv_world_to_grid(x_0, y_0)
        x_end, y_end = self.conv_world_to_grid(x_1, y_1)

        # Check bounds
        if (x_start < 0 or x_start >= self.x_max_grid or y_start < 0 or y_start >= self.y_max_grid or
            x_end < 0 or x_end >= self.x_max_grid or y_end < 0 or y_end >= self.y_max_grid):
            return

        # Setup for Bresenham's line algorithm
        dx = x_end - x_start
        dy = y_end - y_start
        if abs (dy) > abs(dx):
            is_steep = 1
        else:
            is_steep = 0
        if is_steep == 1:
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end

        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start

        dx = x_end - x_start
        dy = y_end - y_start
        error = int(dx / 2.0)
        y_step = 1 if y_start < y_end else -1
        
        # Perform drawing using Bresenham's line algorithm
        y = y_start
        grid_view = self.grid  # Cast to a memoryview before loop to avoid repeated casting
        for x in range(x_start, x_end + 1):
            if is_steep == 1:
                if 0 <= y < self.x_max_grid and 0 <= x < self.y_max_grid:
                    grid_view[y, x] += val
            else:
                if 0 <= x < self.x_max_grid and 0 <= y < self.y_max_grid:
                    grid_view[x, y] += val
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def add_points(self, cnp.ndarray[DTYPE_t, ndim=1] points_x, cnp.ndarray[DTYPE_t, ndim=1] points_y, DTYPE_t val):
        """
        Add a value to an array of points, input coordinates in meters
        points_x, points_y: arrays of x and y coordinates in m
        val: value to add to the cells of the points
        """
        cdef int num_points = points_x.shape[0]
        cdef int i, x_px, y_px
        cdef DTYPE_t[:, :] grid_view = self.grid
        cdef int x_max_grid = self.x_max_grid
        cdef int y_max_grid = self.y_max_grid
        cdef float resolution = self.resolution
        cdef float half_width = self.size_area_world[0] / 2.0
        cdef float half_height = self.size_area_world[1] / 2.0
        
        for i in range(num_points):
            # Convert world coordinates to grid coordinates manually
            x_px = <int>((points_x[i] + half_width) / resolution)
            y_px = <int>((-points_y[i] + half_height) / resolution)

            # Bound checking and adding value
            if 0 <= x_px < x_max_grid and 0 <= y_px < y_max_grid:
                grid_view[x_px, y_px] += val
    
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def add_point(self, float x, float y, DTYPE_t val):
        """
        Add a value to a point, input coordinates in meters
        x, y :  x and y coordinates in m
        val :  value to add to the cell of the point
        """
        cdef int x_px, y_px
        cdef float half_width = self.size_area_world[0] / 2.0
        cdef float half_height = self.size_area_world[1] / 2.0
        cdef DTYPE_t[:, :] grid_view = self.grid
        cdef int x_max_grid = self.x_max_grid
        cdef int y_max_grid = self.y_max_grid
        cdef float resolution = self.resolution

        # Convert world coordinates to grid coordinates manually
        x_px = <int>((x + half_width) / resolution)
        y_px = <int>((-y + half_height) / resolution)

        # Check bounds and add value to the grid
        if 0 <= x_px < x_max_grid and 0 <= y_px < y_max_grid:
            grid_view[x_px, y_px] += val

    def get_grid(self):
        return self.grid
    
    def set_grid(self, cnp.ndarray[DTYPE_t, ndim=2] grid):
        self.grid = grid

    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    cdef add_value_along_line_confidence(self, float x_0, float y_0, float x_1, float y_1, DTYPE_t val):
        cdef int x_start, y_start, x_end, y_end, dx, dy, is_steep, y_step, y, x
        cdef float error, added_value, inverse_dist
        cdef int dist_from_start
        cdef DTYPE_t[:, :] grid_view  # Assuming DTYPE_t is defined as a float or double

        # Handle NaNs
        if isnan(x_0) or isnan(y_0) or isnan(x_1) or isnan(y_1):
            return

        # Convert to grid coordinates and other setup...
        # Convert to grid coordinates
        x_start, y_start = self.conv_world_to_grid(x_0, y_0)
        x_end, y_end = self.conv_world_to_grid(x_1, y_1)

        # Check bounds
        if (x_start < 0 or x_start >= self.x_max_grid or y_start < 0 or y_start >= self.y_max_grid or
            x_end < 0 or x_end >= self.x_max_grid or y_end < 0 or y_end >= self.y_max_grid):
            return

        # Setup for Bresenham's line algorithm
        dx = x_end - x_start
        dy = y_end - y_start
        if abs (dy) > abs(dx):
            is_steep = 1
        else:
            is_steep = 0
        if is_steep == 1:
            x_start, y_start = y_start, x_start
            x_end, y_end = y_end, x_end

        if x_start > x_end:
            x_start, x_end = x_end, x_start
            y_start, y_end = y_end, y_start

        dx = x_end - x_start
        dy = y_end - y_start
        error = int(dx / 2.0)
        y_step = 1 if y_start < y_end else -1

        # Perform drawing using Bresenham's line algorithm
        y = y_start
        grid_view = self.grid  # Cast to a memoryview before loop to avoid repeated casting
        for x in range(x_start, x_end + 1):
            # Calculate the Manhattan distance and cast to int explicitly
            dist_from_start = int(abs(x - x_0) + abs(y - y_0))
            
            # Inline max function and calculate inverse to avoid division
            inverse_dist = 1.0 / float(max(1, dist_from_start/10))
            added_value = val * inverse_dist
            
            # Check boundaries and add value
            if is_steep == 1:
                if 0 <= y < self.x_max_grid and 0 <= x < self.y_max_grid:
                    grid_view[y, x] += added_value
            else:
                if 0 <= x < self.x_max_grid and 0 <= y < self.y_max_grid:
                    grid_view[x, y] += added_value
                
            error -= abs(dy)
            if error < 0:
                y += y_step
                error += dx
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def add_value_along_lines_confidence(self, float x_0, float y_0, cnp.ndarray[DTYPE_t, ndim=1] points_x, cnp.ndarray[DTYPE_t, ndim=1] points_y, DTYPE_t val):
        cdef int num_points = points_x.shape[0]
        cdef int i

        for i in range(num_points):
            self.add_value_along_line_confidence(x_0, y_0, points_x[i], points_y[i], val)
        
    @cython.boundscheck(False)
    @cython.wraparound(False)
    @cython.cdivision(True)
    def add_value_along_lines(self, float x_0, float y_0, cnp.ndarray[DTYPE_t, ndim=1] points_x, cnp.ndarray[DTYPE_t, ndim=1] points_y, DTYPE_t val):
        cdef int num_points = points_x.shape[0]
        cdef int i

        for i in range(num_points):
            self.add_value_along_line(x_0, y_0, points_x[i], points_y[i], val)


cdef class MerkleTree:
    cdef public tuple arr_shape
    cdef public Grid confidence, occupancy
    cdef public object shape
    cdef public int block_size, height, size
    cdef public object tree
    cdef public object differences

    def __cinit__(self, confidence:Grid, occupancy:Grid,  block_size:int = 32) -> None:

        self.arr_shape = confidence.get_grid().shape
        if len(self.arr_shape) != 2:
            raise ValueError("Merkle Tree array must be 2d")

        self.confidence = confidence
        self.occupancy = occupancy
        self.block_size = block_size

        self.shape = np.ceil(np.array(confidence.get_grid().shape) / block_size).astype(int)
        self.height = int(np.max(np.ceil(np.log2(self.shape)))+1)
        self.size = int(4**(self.height-1))
        self.tree = np.zeros(int((4**(self.height)-1)//3), dtype=np.int64)

        self.build(0,self.height, 0, 0)
        self.differences = []

    cdef build(self, node, height, i0, j0):
        self.tree[node] = 0

        if height == 1:
            if i0>=self.arr_shape[0] or j0>=self.arr_shape[1]:
                self.tree[node] = 0
            else:
                i1 = min(i0 + self.block_size, self.arr_shape[0])
                j1 = min(j0 + self.block_size, self.arr_shape[1])
                self.tree[node] = hash(self.confidence.get_grid()[i0:i1, j0:j1].tobytes()) % (1 << 63)

        else:
            for k in range(1,5):
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.build(4 * node + k, height - 1, i, j)
            self.tree[node] = hash(tuple(self.tree[4*node+i] for i in range(1, 5))) % (1 << 63)

    cdef compare_aux(self, other, node, height, i0, j0):
        if self.tree[node] == other.tree[node]:
            return

        if height == 1:
            i1 = i0 + self.block_size
            j1 = j0 + self.block_size
            i1 = int(min(i1, self.arr_shape[0]))
            j1 = int(min(j1, self.arr_shape[1]))
            self.differences.append((i0, j0, i1, j1))
            return

        all_different = True
        for k in range(1,5):
            if self.tree[4*node+k] == other.tree[4*node+k]:
                all_different = False
                break

        if all_different:
            i1 = i0 + self.block_size * (1 << (height - 1))
            j1 = j0 + self.block_size * (1 << (height - 1))
            i1 = int(min(i1, self.arr_shape[0]))
            j1 = int(min(j1, self.arr_shape[1]))
            self.differences.append((i0,j0,i1,j1))
            return

        for k in range(1,5):
            if self.tree[4*node+k] != other.tree[4*node+k]:
                i = i0
                j = j0
                if k >= 3: i += self.block_size * (1 << (height - 2))
                if k % 2 == 0: j += self.block_size * (1 << (height - 2))
                self.compare_aux(other, 4*node+k, height-1, int(i), int(j))

    cdef compare(self, other):
        """
        Returns the rectangles (i0,j0,i1,j1) which differ between the 2 trees
        """
        self.differences = []
        self.compare_aux(other, 0, self.height, 0 ,0)
        return self.differences


    cdef update_aux(self, node, height, curr_rect, rect):
        if height == 1:
            i = min(curr_rect[0] + self.block_size, self.arr_shape[0])
            j = min(curr_rect[1] + self.block_size, self.arr_shape[1])
            self.tree[node] = hash(self.confidence.get_grid()[curr_rect[0]:i, curr_rect[1]:j].tobytes()) % (1 << 63)
            return

        for k in range(1, 5):
            x0,x1 = curr_rect[0],curr_rect[2]
            y0,y1 = curr_rect[1],curr_rect[3]
            if k >= 3:
                x0 += self.block_size * (1 << (height - 2))
            else:
                x1 -= self.block_size * (1 << (height - 2))
            if k % 2 == 0:
                y0 += self.block_size * (1 << (height - 2))
            else:
                y1 -= self.block_size * (1 << (height - 2))

            if (rect[0]<=x0<=rect[2] or rect[0]<=x1<=rect[2]) and (rect[1]<=y0<=rect[3] or rect[1]<=y1<=rect[3]):
                self.update_aux(4 * node + k, height - 1, (x0,y1,x1,y1), rect)

        self.tree[node] = hash(tuple(self.tree[4 * node + i] for i in range(1, 5))) % (1 << 63)


    def update(self, i0, j0, i1, j1):
        """
        updates the merkle tree in the rect (inclusive) specified
        """
        self.update_aux(0, self.height, (0, 0, self.arr_shape[0]-1, self.arr_shape[1]-1), (i0, j0, i1, j1))


    def merge(self, other: 'MerkleTree') -> None:
        differences:List[Tuple[int,int,int,int]] = self.compare(other)
        for i0,j0,i1,j1 in differences:
            self.occupancy.get_grid()[i0:i1, j0:j1] = np.where(self.confidence.get_grid()[i0:i1, j0:j1] > other.confidence.get_grid()[i0:i1, j0:j1], self.occupancy.get_grid()[i0:i1, j0:j1], other.occupancy.get_grid()[i0:i1, j0:j1])
            self.confidence.get_grid()[i0:i1,j0:j1] = np.maximum(self.confidence.get_grid()[i0:i1,j0:j1], other.confidence.get_grid()[i0:i1,j0:j1])
            self.update(i0, j0, i1, j1)
