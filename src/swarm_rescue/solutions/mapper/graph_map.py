from typing import Deque, Tuple, List, Set
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2

FREE_TILE = 0
OBSTACLE_TILE = 1
MAX_CELL_SIZE = 500
MAX_CELL_RADIUS = 30

def get_neighbours(i0:int,j0:int,labels_map,obstacle_map):
    return [
        (i,j) for i in range(i0-1,i0+2) for j in range(j0-1,j0+2)
        if (0<=i<labels_map.shape[0] and
            0<=j<labels_map.shape[1] and
            (i!=i0 or j!=j0) and
            labels_map[i,j]==0 and
            obstacle_map[i,j]==FREE_TILE
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


def create_graph_map(i0,j0,obstacle_map):
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

    return labels_map, total_cell_tiles

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

    labels_map, total_cell_tiles = create_graph_map(60,200,obstacle_map)
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



