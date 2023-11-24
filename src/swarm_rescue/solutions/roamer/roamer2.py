import numpy as np
from scipy.ndimage import morphology
from enum import Enum

class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class FrontierExplorer():

    def __init__(self, map, drone_position):
        self.map = map
        self.drone_position = drone_position

    def computeWFD(self):
        if self.drone_position is None:
            raise ValueError("Robot position not set.")

        # TODO: Copy the occupancy grid into a temporary variable
        temp_map = np.copy(self.map)

        # TODO: Inflate the obstacles if needed

        # Run the Wavefront Detector algorithm
        wavefront_map = np.copy(temp_map)
        wavefront_value = 1
        wavefront_map[self.drone_position] = wavefront_value

        while wavefront_value <= np.max(wavefront_map):
            indices = np.where(wavefront_map == wavefront_value)

            for i, j in zip(indices[0], indices[1]):
                neighbors = [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

                for neighbor in neighbors:
                    if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1] and wavefront_map[neighbor] == Zone.EMPTY.value:
                        wavefront_map[neighbor] = wavefront_value + 1

            wavefront_value += 1

        print(wavefront_map)
        # Extract frontiers from the computed wavefront map
        frontiers = self.extract_frontiers(wavefront_map)

        return frontiers

    def extract_frontiers(self, wavefront_map):
        # Extract frontiers from the wavefront map
        frontiers = []
        visited = set()

        for i in range(wavefront_map.shape[0]):
            for j in range(wavefront_map.shape[1]):
                if wavefront_map[i, j] == Zone.INEXPLORED.value and (i, j) not in visited:
                    frontier = self.find_frontier(wavefront_map, i, j, visited)
                    if len(frontier) > 0:
                        frontiers.append(frontier)

        return frontiers

    def find_frontier(self, wavefront_map, start_row, start_col, visited):
        # DFS to find a frontier starting from the given point
        frontier = []
        stack = [(start_row, start_col)]

        while stack:
            current_row, current_col = stack.pop()

            has_empty_neighbor = any(
                    wavefront_map[n_row, n_col] > 1
                    for n_row, n_col in [
                        (current_row - 1, current_col),
                        (current_row + 1, current_col),
                        (current_row, current_col - 1),
                        (current_row, current_col + 1),
                    ]
                    if (
                        0 <= n_row < wavefront_map.shape[0]
                        and 0 <= n_col < wavefront_map.shape[1]
                    )
                )

            if wavefront_map[current_row, current_col] == Zone.INEXPLORED.value and (current_row, current_col) not in visited and has_empty_neighbor:
 
                print("Point ", current_row, current_col, " has empty neighbor : ", has_empty_neighbor)

                frontier.append((current_row, current_col))
                visited.add((current_row, current_col))

                neighbors = [(current_row + x, current_col + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

                for neighbor in neighbors:
                    if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1]:
                        stack.append(neighbor)

        return frontier

    def getRandomFrontier(self, request):
        """ Return random frontier """
        # TODO
        frontiers = self.computeWFD()
        frontier = np.random.choice(frontiers)

        frontierCenter = 0  # TODO: compute center of the randomly drawn frontier here
        x, y = 0, 0  # TODO: transform the coordinates from grid to real-world coordinates (in meters)
        return None

    def getClosestFrontier(self, request):
        """ Return frontier closest to the robot """
        # TODO
        frontiers = self.computeWFD()
        bestFrontierIdx = 0  # TODO: compute the index of the best frontier
        frontier = frontiers[bestFrontierIdx]

        frontierCenter = 0  # TODO: compute the center of the chosen frontier
        x, y = 0, 0  # TODO: compute the index of the best frontier
        return None

    def extractGrid(self, msg):
        # TODO: extract grid from msg.data and other usefull information
        pass


def convert_to_zone(num_array):
    return np.vectorize(lambda x: Zone(x))(num_array)

def print_frontiers_on_map_with_letters(matrix, frontiers):
    frontier_map = np.array(matrix, dtype=str)

    for index, frontier in enumerate(frontiers):
        for point in frontier:
            frontier_map[point] = chr(ord('A') + index)

    for row in frontier_map:
        print(" ".join(row))


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

fd = FrontierExplorer(matrix, start_pos)
frontiers = fd.computeWFD()
print(frontiers)
print_frontiers_on_map_with_letters(matrix, frontiers)
