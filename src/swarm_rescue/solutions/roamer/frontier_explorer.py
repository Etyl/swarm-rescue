from typing import List, Set, Optional, Tuple
import numpy as np
from scipy.ndimage import label # type: ignore
from enum import Enum

from solutions.utils.types import Vector2D # type: ignore


class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1


def get_frontiers(map, drone_position: Vector2D, frontiers_threshold: int) -> Tuple[List[List[Vector2D]],Optional[List[int]]]:
    """
    Compute the Wavefront Detector algorithm on the current map
    """
    if drone_position is None:
        raise ValueError("Robot position not set.")

    temp_map = np.copy(map)

    # Run the Wavefront Detector algorithm
    wavefront_map = np.copy(temp_map)
    wavefront_value = 1
    wavefront_map[drone_position.x, drone_position.y] = wavefront_value

    while wavefront_value <= np.max(wavefront_map):
        indices = np.where(wavefront_map == wavefront_value)

        for i, j in zip(indices[0], indices[1]):
            neighbors = [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

            for neighbor in neighbors:
                if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1] and wavefront_map[neighbor] == Zone.EMPTY.value:
                    wavefront_map[neighbor] = wavefront_value + 1

        wavefront_value += 1

    # Extract frontiers from the computed wavefront map
    frontiers = extract_frontiers(wavefront_map, frontiers_threshold)

    connectedMap, connectedCount = buildConnectedComponents(map)
    frontier_count = getFrontiersCount(frontiers, connectedMap, connectedCount)

    return frontiers, frontier_count

def extract_frontiers(wavefront_map, min_points=5) -> List[List[Vector2D]]:
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
                frontier = find_frontier(wavefront_map, i, j, visited)
                if len(frontier) > min_points:
                    frontiers.append(frontier)

    return frontiers

def find_frontier(wavefront_map, start_row, start_col, visited) -> List[Vector2D]:
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


def buildConnectedComponents(input_map):
    """
    Build the connected components of the given map
    """
    map = np.abs(input_map[2:-2, 2:-2])
    # Build the connected components of the map

    structure = np.ones((3, 3), dtype=int)
    connectedMap, ncomponents = label(map, structure)

    values, connectedCount = np.unique(connectedMap, return_counts=True)

    fullConnectedMap = ncomponents*np.ones(input_map.shape, dtype=int)
    fullConnectedMap[2:-2, 2:-2] = connectedMap

    return fullConnectedMap, connectedCount



def getFrontiersCount(frontiers, connectedMap, connectedCount) -> Optional[List[int]]:
    """
    Return the frontiers count
    """
    if connectedMap is None or connectedCount is None:
        return None

    frontier_count : List[int] = []
    for frontier in frontiers:

        center = frontier[len(frontier)//2]
        connected_zone_id = connectedMap[center.x,center.y]
        frontier_count.append(connectedCount[connected_zone_id])

    return frontier_count

