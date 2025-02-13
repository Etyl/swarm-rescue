import heapq
from typing import Deque, Tuple, List, Set, Dict
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

FREE_TILE = 0
OBSTACLE_TILE = 1
UNKNOWN_TILE = 2
MAX_CELL_SIZE = 500
MAX_CELL_RADIUS = 30

def get_neighbours(i0:int,j0:int,labels_map,obstacle_map, include_unknown=False):
    return [
        (i,j) for i in range(i0-1,i0+2) for j in range(j0-1,j0+2)
        if (0<=i<labels_map.shape[0] and
            0<=j<labels_map.shape[1] and
            (i!=i0 or j!=j0) and
            (include_unknown or labels_map[i,j]==0) and
            (obstacle_map[i,j]==FREE_TILE or (include_unknown and obstacle_map[i,j]==UNKNOWN_TILE))
        )
    ]

def propagate_cell(i0:int,j0:int,labels_map,obstacle_map,label) \
    -> Tuple[Set[Tuple[int,int]],List[Tuple[int,int]]]:
    """
    Propagates a cell starting at i0,j0 to surrounding neighbours until it has reached max size
    Returns remaining neighbours
    """
    wait : Deque[Tuple[int,int]] = deque([(i0,j0)])
    cell_tiles : List[Tuple[int,int]] = []
    cell_size = 0
    min_i,max_i = i0,i0
    min_j,max_j = j0,j0
    while len(wait) > 0 and cell_size<MAX_CELL_SIZE and max_j-min_j<=MAX_CELL_RADIUS and max_i-min_i<=MAX_CELL_RADIUS:
        i,j = wait.popleft()
        if labels_map[i,j]!=0:
            continue
        min_i,max_i = min(min_i,i),max(max_i,i)
        min_j,max_j = min(min_j,j),max(max_j,j)
        cell_tiles.append((i,j))
        labels_map[i, j] = label
        cell_size += 1
        for neigh in get_neighbours(i,j,labels_map,obstacle_map):
            wait.append(neigh)

    return set(wait), cell_tiles


def get_features(labels_map, total_cell_tiles, obstacle_map):
    features = []
    A = np.zeros((len(total_cell_tiles),len(total_cell_tiles)))
    for k in range(len(total_cell_tiles)):
        label = k+1
        perimeter = 0
        barycenter = [0,0]
        frontier = set()
        for l,(i0,j0) in enumerate(total_cell_tiles[k]):
            barycenter[0] += i0
            barycenter[1] += j0
            for i,j in get_neighbours(i0,j0,labels_map,obstacle_map, include_unknown=True):
                if labels_map[i,j]!=label and obstacle_map[i,j]==FREE_TILE:
                    A[label-1, labels_map[i,j]-1] = 1
                    A[labels_map[i,j]-1, label-1] = 1
                    perimeter += 1
                if obstacle_map[i,j]==UNKNOWN_TILE:
                    frontier.add((i0,j0))

        barycenter[0] /= len(total_cell_tiles[k])
        barycenter[1] /= len(total_cell_tiles[k])
        frontier_barycenter = [barycenter[0],barycenter[1]]
        if len(frontier) > 0:
            frontier_barycenter[0] = sum([p[0] for p in frontier]) / len(frontier)
            frontier_barycenter[1] = sum([p[1] for p in frontier]) / len(frontier)

        node_features = dict()
        node_features["area"] = len(total_cell_tiles[k])
        node_features["perimeter"] = perimeter
        node_features["barycenter_x"] = barycenter[0]
        node_features["barycenter_y"] = barycenter[1]
        node_features["frontier_size"] = len(frontier)
        node_features["frontier_barycenter_x"] = frontier_barycenter[0]
        node_features["frontier_barycenter_y"] = frontier_barycenter[1]
        features.append(node_features)

    return features, A


def add_drone_distance(A, features, starting_node):
    num_nodes = len(features)
    dist = 100_000_000 * np.ones(num_nodes)  # Initialize distances with a large value
    dist[starting_node] = 0  # Distance to itself is zero
    visited = np.zeros(len(features)).astype(bool)  # Track visited nodes
    priority_queue = [(0, starting_node)]  # Min-heap (distance, node)

    while priority_queue:
        current_dist, current_node = heapq.heappop(priority_queue)

        if visited[current_node]:
            continue
        visited[current_node] = True

        for neighbor in range(num_nodes):
            if A[current_node, neighbor] > 0:
                current_node_pos = features[current_node]["barycenter_x"],features[current_node]["barycenter_y"]
                neighbor_pos = features[neighbor]["barycenter_x"],features[neighbor]["barycenter_y"]
                new_dist = current_dist + np.sqrt(
                    (current_node_pos[0]-neighbor_pos[0])**2
                    +(current_node_pos[1]-neighbor_pos[1])**2
                )
                if new_dist < dist[neighbor]:
                    dist[neighbor] = new_dist
                    heapq.heappush(priority_queue, (new_dist, neighbor))

    for k in range(num_nodes):
        features[k]["drone_distance"] = dist[k]


def create_graph_map(i0,j0,obstacle_map:np.ndarray)\
    -> Tuple[
        np.ndarray,
        List[List[Tuple[int,int]]],
        List[Dict[str,float]],
        np.ndarray
    ]:
    """
    Params:
        i0,j0: Coords of the drone
        obstacle map: map with FREE_TILE, OBSTACLE_TILE, UNKNOWN_TILE
    Returns:
        labels_map: label[i,j] -> id of the corresponding node
        total_cell_tiles: total_cell_tiles[node_id] -> list of all points associated to this node
        features: features[node_id] -> dict of the features for this node
        A: adjacency matrix of the graph
    """
    labels_map = np.zeros_like(obstacle_map)
    wait = {(i0, j0)}
    current_label = 1
    total_cell_tiles = []
    while len(wait) > 0:
        i,j = wait.pop()
        if labels_map[i,j] != 0:
            continue
        new_wait, cell_tiles = propagate_cell(i,j,labels_map,obstacle_map,current_label)
        current_label += 1
        total_cell_tiles.append(cell_tiles)
        wait = wait | new_wait

    features, A = get_features(labels_map, total_cell_tiles, obstacle_map)
    drone_node = labels_map[i0,j0]
    add_drone_distance(A, features, drone_node)

    return labels_map, total_cell_tiles, features, A

COLORS = [
    (255, 0, 0),    # Red
    (0, 255, 0),    # Green
    (0, 0, 255),    # Blue
    (255, 255, 0),  # Yellow
    (255, 165, 0),  # Orange
    (128, 0, 128),  # Purple
    (0, 255, 255),  # Cyan
    (255, 192, 203),# Pink
    (165, 42, 42),  # Brown
]

if __name__ == '__main__':
    image = mpimg.imread("obstacle.png")
    image = cv2.resize(image, (230, 170))

    obstacle_map = np.mean(image, axis=2)
    obstacle_map = np.where(obstacle_map < 0.8, 1, 0)

    labels_map, total_cell_tiles, _ , _ = create_graph_map(60,200,obstacle_map)
    graph_display = np.zeros_like(image)
    for k in range(len(total_cell_tiles)):
        for i,j in total_cell_tiles[k]:
            graph_display[i,j] = COLORS[k%len(COLORS)]
    graph_display /= 255
    plt.figure(figsize=(10, 10))
    plt.imshow(obstacle_map)
    plt.figure()
    plt.imshow(graph_display)
    plt.show()



