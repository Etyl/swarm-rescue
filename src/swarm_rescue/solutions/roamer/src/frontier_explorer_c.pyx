import numpy as np
from scipy.ndimage import label # type: ignore
import matplotlib.pyplot as plt

#from libcpp.set cimport set
#from libcpp.pair cimport pair
#from libcpp.vector cimport vector
cimport numpy as cnp
cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t


ctypedef enum Zone:
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1
    NEG_OBSTACLE = -2


cdef find_frontier(wavefront_map, int start_row,int start_col, visited):
    """
    Find a frontier starting from the given point
    Doing a DFS to find a frontier starting from the given point
    and itering over the neighbors to find the frontier by spreading
    """

    # DFS to find a frontier starting from the given point
    frontier = []
    stack = [(start_row, start_col)]

    while len(stack)>0:
        (current_row, current_col) = stack.pop()

        if wavefront_map[current_row, current_col] != INEXPLORED or visited[current_row, current_col]:
            continue

        has_empty_neighbor = False
        for n_row, n_col in [
            (current_row - 1, current_col),
            (current_row + 1, current_col),
            (current_row, current_col - 1),
            (current_row, current_col + 1)]:
            if (
                0 <= n_row < wavefront_map.shape[0]
                and 0 <= n_col < wavefront_map.shape[1]
                and wavefront_map[n_row, n_col] > 1):
                has_empty_neighbor = True
                break


        if not has_empty_neighbor:
            continue

        frontier.append([current_row, current_col])
        visited[current_row, current_col] = True

        neighbors = [(current_row + x, current_col + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

        for (i,j) in neighbors:
            if 0 <= i < wavefront_map.shape[0] and 0 <= j < wavefront_map.shape[1] and not visited[i,j] and wavefront_map[i,j]==INEXPLORED:
                stack.append((i,j))

    return frontier

cdef extract_frontiers(wavefront_map, int min_points=5):
    """
    Extract frontiers from the wavefront map
    params:
        - wavefront_map: the wavefront map
        - min_points: the minimum number of points in a frontier
    """
    # Extract frontiers from the wavefront map
    frontiers = []
    cdef cnp.ndarray[cnp.int8_t, ndim=2, cast=True] visited = np.zeros(wavefront_map.shape, dtype=bool)

    for i in range(wavefront_map.shape[0]):
        for j in range(wavefront_map.shape[1]):
            if wavefront_map[i, j] == INEXPLORED and not visited[i,j]:
                frontier = find_frontier(wavefront_map, i, j, visited)
                if len(frontier) > min_points:
                    frontiers.append(frontier)

    return frontiers

cdef buildConnectedComponents(input_map):
    """
    Build the connected components of the given map
    """
    cdef cnp.ndarray[DTYPE_t, ndim=2] map = np.abs(input_map[2:-2, 2:-2]).astype(DTYPE)
    # Build the connected components of the map

    structure = np.ones((3, 3), dtype=int)
    connectedMap, ncomponents = label(map, structure)

    values, connectedCount = np.unique(connectedMap, return_counts=True)

    fullConnectedMap = ncomponents*np.ones(input_map.shape, dtype=int)
    fullConnectedMap[2:-2, 2:-2] = connectedMap

    return fullConnectedMap, connectedCount


cdef getFrontiersCount(frontiers, connectedMap, connectedCount):
    """
    Return the frontiers count
    """
    if connectedMap is None or connectedCount is None:
        return None

    frontier_count = []
    for frontier in frontiers:

        center = frontier[len(frontier)//2]
        connected_zone_id = connectedMap[center[0],center[1]]
        frontier_count.append(connectedCount[connected_zone_id])

    return frontier_count

def get_frontiers(map, drone_position, frontiers_threshold: int):
    """
    Compute the Wavefront Detector algorithm on the current map
    """
    # Run the Wavefront Detector algorithm
    cdef cnp.ndarray[DTYPE_t, ndim=2] wavefront_map = map.copy().astype(DTYPE)
    wavefront_map[wavefront_map==OBSTACLE] = NEG_OBSTACLE
    wavefront_value = 1
    wavefront_map[drone_position[0], drone_position[1]] = wavefront_value

    while wavefront_value <= np.max(wavefront_map):
        indices = np.where(wavefront_map == wavefront_value)

        for i, j in zip(indices[0], indices[1]):
            neighbors = [(i + x, j + y) for x in [-1, 0, 1] for y in [-1, 0, 1] if (x != 0 or y != 0)]

            for neighbor in neighbors:
                if 0 <= neighbor[0] < wavefront_map.shape[0] and 0 <= neighbor[1] < wavefront_map.shape[1] and wavefront_map[neighbor] == EMPTY:
                    wavefront_map[neighbor] = wavefront_value + 1

        wavefront_value += 1

    # plt.imsave("map.png",map)
    # plt.imsave("wavefront.png",wavefront_map)

    # Extract frontiers from the computed wavefront map
    frontiers = extract_frontiers(wavefront_map, frontiers_threshold)

    connectedMap, connectedCount = buildConnectedComponents(map)
    frontier_count = getFrontiersCount(frontiers, connectedMap, connectedCount)

    return frontiers, frontier_count
