from typing import List, Set, Optional
import numpy as np
from scipy.ndimage import label # type: ignore
from enum import Enum

from solutions.types.types import Vector2D # type: ignore


class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class FrontierExplorer:
    """
    Class that implements the Wavefront Detector algorithm
    """
    def __init__(self, map, drone_position: Vector2D, frontiers_threshold: int):
        self.map = map
        self.drone_position : Vector2D = drone_position
        self.frontiers_threshold : int = frontiers_threshold
        self.frontiers : List[List[Vector2D]] = self.computeWFD()
        self.connectedMap : Optional[np.ndarray[int, np.shape(2)]] = None
        self.connectedCount : Optional[List[int]] = None
        self.buildConnectedComponents()

    def computeWFD(self) -> List[List[Vector2D]]:
        """
        Compute the Wavefront Detector algorithm on the current map
        """
        if self.drone_position is None:
            raise ValueError("Robot position not set.")

        temp_map = np.copy(self.map)

        # Run the Wavefront Detector algorithm
        wavefront_map = np.copy(temp_map)
        wavefront_value = 1
        wavefront_map[self.drone_position.x, self.drone_position.y] = wavefront_value

        while wavefront_value <= np.max(wavefront_map):
            indices = np.where(wavefront_map == wavefront_value)

            for i, j in zip(indices[0], indices[1]):
                neighbors = [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

                for neighbor in neighbors:
                    if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1] and wavefront_map[neighbor] == Zone.EMPTY.value:
                        wavefront_map[neighbor] = wavefront_value + 1

            wavefront_value += 1

        # Extract frontiers from the computed wavefront map
        frontiers = self.extract_frontiers(wavefront_map, self.frontiers_threshold)
        
        return frontiers

    def extract_frontiers(self, wavefront_map, min_points=5) -> List[List[Vector2D]]:
        """
        Extract frontiers from the wavefront map
        params:
            - wavefront_map: the wavefront map
            - min_points: the minimum number of points in a frontier
        """
        # Extract frontiers from the wavefront map
        frontiers = []
        visited : Set = set()
        
        for i in range(wavefront_map.shape[0]):
            for j in range(wavefront_map.shape[1]):
                if wavefront_map[i, j] == Zone.INEXPLORED.value and (i, j) not in visited:
                    frontier = self.find_frontier(wavefront_map, i, j, visited)
                    if len(frontier) > min_points:
                        frontiers.append(frontier)

        return frontiers

    def find_frontier(self, wavefront_map, start_row, start_col, visited) -> List[Vector2D]:
        """
        Find a frontier starting from the given point
        Doing a DFS to find a frontier starting from the given point
        and itering over the neighbors to find the frontier by spreading
        """

        # DFS to find a frontier starting from the given point
        frontier : List[Vector2D] = []
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
 
                # DEBUG
                # print("Point ", current_row, current_col, " has empty neighbor : ", has_empty_neighbor)
                # END DEBUG

                frontier.append(Vector2D(current_row, current_col))
                visited.add((current_row, current_col))

                neighbors = [(current_row + x, current_col + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

                for neighbor in neighbors:
                    if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1]:
                        stack.append(neighbor)
        return frontier
    

    def buildConnectedComponents(self):
        """
        Build the connected components of the given map
        """
        map = np.abs(self.map[2:-2, 2:-2])
        # Build the connected components of the map
        
        structure = np.ones((3, 3), dtype=int)
        connectedMap, ncomponents = label(map, structure)

        values, connectedCount = np.unique(connectedMap, return_counts=True)

        fullConnectedMap = ncomponents*np.ones(self.map.shape, dtype=int)
        fullConnectedMap[2:-2, 2:-2] = connectedMap
        self.connectedMap = fullConnectedMap

        self.connectedCount = np.append(connectedCount, [0])


    def getClosestFrontier(self):
        """
        Find the frontier with the closest center to the robot
        """
        frontiers = self.frontiers

        if not frontiers:
            return None  # No frontiers found

        curr_row, curr_col = self.drone_position

        # Find the frontier with the closest center to the robot
        best_distance = float('inf')
        best_frontier_idx = 0

        for idx, frontier in enumerate(frontiers):
            # Calculate the center of the frontier
            frontier_center = Vector2D(
                sum(point.x for point in frontier) / len(frontier),
                sum(point.y for point in frontier) / len(frontier)
            )

            # Calculate the distance from the robot to the center of the frontier
            distance = np.sqrt((frontier_center.x - curr_row)**2 + (frontier_center.y - curr_col)**2)

            # Update the best frontier if the current one is closer
            if distance < best_distance:
                best_distance = distance
                best_frontier_idx = idx

        # Return the center and the points of the chosen frontier
        chosen_frontier = frontiers[best_frontier_idx]
        chosen_frontier_center = (
            int(sum(point.x for point in chosen_frontier) / len(chosen_frontier)),
            int(sum(point.y for point in chosen_frontier) / len(chosen_frontier))
        )

        return chosen_frontier_center

    def getFrontiers(self) -> List[List[Vector2D]]:
        """
        Return the frontiers
        """
        return self.frontiers
    
    def getFrontiersCount(self) -> Optional[List[int]]:
        """
        Return the frontiers count
        """
        if self.connectedMap is None or self.connectedCount is None:
            return None

        frontier_count : List[int] = []
        for frontier in self.frontiers:

            center = frontier[len(frontier)//2]
            connected_zone_id = self.connectedMap[center.x,center.y]
            frontier_count.append(self.connectedCount[connected_zone_id])
        
        return frontier_count


    def extractGrid(self, msg):
        # TODO: extract grid from msg.data and other usefull information
        pass

