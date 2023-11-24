import numpy as np
from enum import Enum
import os
import pyastar2d
from scipy.ndimage import binary_dilation
from scipy.ndimage import convolve

class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1
    TESTED = 4

def find_next_unexeplored_target(map, position):
        """
        Find the closest unexplored target from the drone's curretn position
        It comes to finding the closest INEXPLORED point which is next to a explored point in the map
        """

        rows, cols = map.shape
        current_row, current_col = position

        def is_valid_move(row, col):
            return 0 <= row < rows and 0 <= col < cols and map[row, col] != Zone.OBSTACLE

        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Helper function to check if a cell has a neighbor with Zone value >= 0
        def has_valid_neighbor(row, col):
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if is_valid_move(new_row, new_col) and map[new_row, new_col].value >= 0:
                    return True
            return False

        # BFS to find the closest unexplored point with a valid neighbor
        found_point = None
        queue = [(current_row, current_col)]
        visited = set()

        while queue:
            current_row, current_col = queue.pop(0)

            if map[current_row, current_col] == Zone.INEXPLORED and has_valid_neighbor(current_row, current_col):
                print(f"Found unexplored target at {current_row, current_col}")
                found_point = np.array([current_row, current_col])
                break
            
            # map[current_row, current_col] = Zone.TESTED
            print_num_map(map)

            visited.add((current_row, current_col))

            for dr, dc in moves:
                new_row, new_col = current_row + dr, current_col + dc
                if is_valid_move(new_row, new_col) and (new_row, new_col) not in visited:
                    queue.append((new_row, new_col))

        return found_point

def print_num_map(map, output_file='output.txt'):
    numeric_map = np.vectorize(lambda x: x.value)(map)
    
    # Save to file with alignment
    script_dir = os.path.dirname(os.path.abspath(__file__))
    output_path = os.path.join(script_dir, output_file)

    with open(output_path, 'a') as file:
        np.savetxt(file, numeric_map, fmt='%3d', delimiter=', ')
        file.write('\n') 

def convert_to_zone(num_array):
    return np.vectorize(lambda x: Zone(x))(num_array)

def convert_matrix_for_astar(matrix):
    INF = 1000

    conversion_dict = {
        Zone.INEXPLORED: INF,
        Zone.OBSTACLE: INF,
        Zone.RESCUE_CENTER: 1,
        Zone.WOUNDED: 1,
        Zone.EMPTY: 1,
        Zone.TESTED: 1
    }

    converted_matrix = np.vectorize(lambda x: conversion_dict[x])(matrix)
    return converted_matrix.astype(np.float32)

def print_frontiers_on_map_with_letters(matrix, frontiers):
    frontier_map = np.array(matrix, dtype=str)

    for index, frontier in enumerate(frontiers):
        for point in frontier:
            frontier_map[point] = chr(ord('A') + index)

    for row in frontier_map:
        print(" ".join(row))

def thicken_walls(original_map, wall_thickness=1):
    # Create a binary map where obstacles are considered as foreground (1) and others as background (0)
    binary_map = (original_map == Zone.OBSTACLE.value).astype(int)

    # Perform morphological dilation to thicken the walls
    dilated_map = binary_dilation(binary_map, iterations=wall_thickness)

    # Convert the dilated binary map back to the numerical map
    thickened_map = original_map.copy()
    thickened_map[dilated_map == 1] = Zone.OBSTACLE.value

    return thickened_map

def thicken_walls2(original_map, n=1):
    obstacle_map = (original_map == Zone.OBSTACLE.value).astype(int)

    # Define a 3x3 kernel for convolution
    kernel = np.ones((3, 3), dtype=int)

    # Use convolution to count obstacle neighbors
    neighbor_count = convolve(obstacle_map, kernel, mode='constant', cval=0)

    # Set cells as obstacles if they have at least n obstacle neighbors
    thickened_map = original_map.copy()
    thickened_map[neighbor_count >= n] = Zone.OBSTACLE.value

    return thickened_map
    
def thicken_walls3(matrix, n):
    new_matrix = np.copy(matrix)
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == Zone.INEXPLORED.value:
                for x in range(max(0, i - n), min(rows, i + n + 1)):
                    for y in range(max(0, j - n), min(cols, j + n + 1)):
                        if matrix[x, y] == Zone.EMPTY.value:
                            new_matrix[x, y] = Zone.OBSTACLE.value

    return new_matrix

def thicken_walls4(matrix, n):
    new_matrix = np.copy(matrix)
    rows, cols = matrix.shape

    for i in range(rows):
        for j in range(cols):
            if matrix[i, j] == Zone.OBSTACLE:
                for x in range(max(0, i - n), min(rows, i + n + 1)):
                    for y in range(max(0, j - n), min(cols, j + n + 1)):
                        if matrix[x, y] == Zone.EMPTY:
                            new_matrix[x, y] = Zone.OBSTACLE

    return new_matrix

start_pos = np.array([3, 4])
matrix = np.array([
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1],
    [-1,  1,  1,  1,  1,  1,  1, -1, -1, -1],
    [-1,  0,  0,  1,  0,  0,  1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  0,  1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  0,  1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  0,  1, -1, -1, -1],
    [-1,  0,  0,  0,  0,  0,  1, -1, -1, -1],
    [-1, -1, -1,  0,  0,  0,  0, -1, -1, -1],
    [-1, -1, -1,  0,  0,  0,  0, -1, -1, -1],
    [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
])

# matrix_zone = convert_to_zone(matrix)
# found_point = find_next_unexeplored_target(matrix_zone, start_pos)
# matrix_astar = convert_matrix_for_astar(matrix_zone)
# path = pyastar2d.astar_path(matrix_astar, tuple(start_pos), tuple(found_point), allow_diagonal=True)

# fd = FrontierExplorer(matrix, start_pos)
# frontiers = fd.computeWFD()
# print_frontiers_on_map_with_letters(matrix, frontiers)
# print(fd.getClosestFrontier())

matrix_zone = convert_to_zone(matrix)
matrix_astar1 = convert_matrix_for_astar(matrix_zone)

matrix_zone = convert_to_zone(matrix)
thickened_map = thicken_walls4(matrix_zone, 1)
print(thickened_map == matrix_zone)
matrix_astar2 = convert_matrix_for_astar(thickened_map)
print(matrix_astar1)
print(matrix_astar2)

