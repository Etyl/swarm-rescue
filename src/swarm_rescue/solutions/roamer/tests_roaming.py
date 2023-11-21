import numpy as np
from enum import Enum
import os
import pyastar2d

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

matrix_zone = convert_to_zone(matrix)
found_point = find_next_unexeplored_target(matrix_zone, start_pos)
matrix_astar = convert_matrix_for_astar(matrix_zone)

path = pyastar2d.astar_path(matrix_astar, tuple(start_pos), tuple(found_point), allow_diagonal=True)



