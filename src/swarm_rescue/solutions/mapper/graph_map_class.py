import heapq
from typing import Deque, Tuple, List, Set, Dict
import numpy as np
from collections import deque
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import cv2
import networkx as nx
import arcade

from solutions.mapper.mapper import Map, Zone  # type: ignore
from solutions.utils.types import Vector2D # type: ignore

FREE_TILE = 0
OBSTACLE_TILE = 1
UNKNOWN_TILE = 2
MAX_CELL_SIZE = 500
MAX_CELL_RADIUS = 30


class GraphMap:
    def __init__(self, drone, map: Map):
        self.drone = drone
        self.map = map
        self.labels_map = np.zeros((map.get_width(), map.get_height()), dtype=int)
        self.total_cell_tiles = []
        self.adjacency_matrix = None
        self.graph = None
        self.features = None

    def get_neighbours(self, i0: int, j0: int, include_unknown=False):
        return [
            (i, j)
            for i in range(i0 - 1, i0 + 2)
            for j in range(j0 - 1, j0 + 2)
            if (
                0 <= i < self.map.get_width()
                and 0 <= j < self.map.get_height()
                and (i != i0 or j != j0)
                and (include_unknown or self.labels_map[i, j] == 0)
                and (
                    self.map[Vector2D(i, j)] == Zone.EMPTY
                    or (include_unknown and self.map[Vector2D(i, j)] == Zone.UNEXPLORED)
                )
            )
        ]

    def propagate_cell(
        self, i0: int, j0: int, label
    ) -> Tuple[Set[Tuple[int, int]], List[Tuple[int, int]]]:
        """
        Propagates a cell starting at i0,j0 to surrounding neighbours until it has reached max size
        Returns remaining neighbours
        """
        wait: Deque[Tuple[int, int]] = deque([(i0, j0)])
        cell_tiles: List[Tuple[int, int]] = []
        cell_size = 0
        min_i, max_i = i0, i0
        min_j, max_j = j0, j0
        while (
            len(wait) > 0
            and cell_size < MAX_CELL_SIZE
            and max_j - min_j <= MAX_CELL_RADIUS
            and max_i - min_i <= MAX_CELL_RADIUS
        ):
            i, j = wait.popleft()
            if self.labels_map[i, j] != 0:
                continue
            min_i, max_i = min(min_i, i), max(max_i, i)
            min_j, max_j = min(min_j, j), max(max_j, j)
            cell_tiles.append((i, j))
            self.labels_map[i, j] = label
            cell_size += 1
            for neigh in self.get_neighbours(i, j):
                wait.append(neigh)

        return set(wait), cell_tiles

    def get_features(self):
        features = []
        A = np.zeros((len(self.total_cell_tiles), len(self.total_cell_tiles)))
        for k in range(len(self.total_cell_tiles)):
            label = k + 1
            perimeter = 0
            barycenter = [0, 0]
            frontier = set()
            for l, (i0, j0) in enumerate(self.total_cell_tiles[k]):
                barycenter[0] += i0
                barycenter[1] += j0
                for i, j in self.get_neighbours(i0, j0, include_unknown=True):
                    if self.labels_map[i, j] != label and self.map[Vector2D(i, j)] == Zone.EMPTY:
                        A[label - 1, self.labels_map[i, j] - 1] = 1
                        A[self.labels_map[i, j] - 1, label - 1] = 1
                        perimeter += 1
                    if self.map[Vector2D(i, j)] == Zone.UNEXPLORED:
                        frontier.add((i0, j0))

            barycenter[0] /= len(self.total_cell_tiles[k])
            barycenter[1] /= len(self.total_cell_tiles[k])
            frontier_barycenter = [barycenter[0], barycenter[1]]
            if len(frontier) > 0:
                frontier_barycenter[0] = sum([p[0] for p in frontier]) / len(frontier)
                frontier_barycenter[1] = sum([p[1] for p in frontier]) / len(frontier)

            node_features = dict()
            node_features["area"] = len(self.total_cell_tiles[k])
            node_features["perimeter"] = perimeter
            node_features["barycenter_x"] = barycenter[0]
            node_features["barycenter_y"] = barycenter[1]
            node_features["frontier_size"] = len(frontier)
            node_features["frontier_barycenter_x"] = frontier_barycenter[0]
            node_features["frontier_barycenter_y"] = frontier_barycenter[1]
            features.append(node_features)

        return features, A

    def add_drone_distance(self, A, features, starting_node):
        num_nodes = len(features)
        dist = 100_000_000 * np.ones(
            num_nodes
        )  # Initialize distances with a large value
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
                    current_node_pos = (
                        features[current_node]["barycenter_x"],
                        features[current_node]["barycenter_y"],
                    )
                    neighbor_pos = (
                        features[neighbor]["barycenter_x"],
                        features[neighbor]["barycenter_y"],
                    )
                    new_dist = current_dist + np.sqrt(
                        (current_node_pos[0] - neighbor_pos[0]) ** 2
                        + (current_node_pos[1] - neighbor_pos[1]) ** 2
                    )
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        heapq.heappush(priority_queue, (new_dist, neighbor))

        for k in range(num_nodes):
            features[k]["drone_distance"] = dist[k]

    def draw(self):
        if self.graph is not None:

            for i in range(len(self.adjacency_matrix)):
                for j in range(i + 1, len(self.adjacency_matrix)):
                    
                    if self.adjacency_matrix[i][j] == 1:
          
                        pos_x_i = self.features[i]["barycenter_x"]
                        pos_y_i = self.features[i]["barycenter_y"]
                        pos_i = Vector2D(pos_x_i, pos_y_i)
                        pos_i = self.map.grid_to_world(pos_i)
                        pos_i = pos_i.array + np.array(self.drone.size_area) / 2

                        pos_x_j = self.features[j]["barycenter_x"]
                        pos_y_j = self.features[j]["barycenter_y"]
                        pos_j = Vector2D(pos_x_j, pos_y_j)
                        pos_j = self.map.grid_to_world(pos_j)
                        pos_j = pos_j.array + np.array(self.drone.size_area) / 2

                        node_size_i = (self.features[i]["area"] / MAX_CELL_SIZE + 1)*5
                        node_size_j = (self.features[j]["area"] / MAX_CELL_SIZE + 1)*5
    
                        perimeter_i = self.features[i]["perimeter"] / (MAX_CELL_RADIUS * 8) 
                        perimeter_j = self.features[j]["perimeter"] / (MAX_CELL_RADIUS * 8)

                        # Gradient of color depending on the frontier size 
                        color_start = arcade.color.BLACK
                        color_end = arcade.color.RED
                        t_i = min(self.features[i]["frontier_size"] / 10, 1)
                        t_j = min(self.features[j]["frontier_size"] / 10, 1)
                        node_color_i = (
                            color_start[0] + t_i * (color_end[0] - color_start[0]),
                            color_start[1] + t_i * (color_end[1] - color_start[1]),
                            color_start[2] + t_i * (color_end[2] - color_start[2]),
                        )
                        node_color_j = (
                            color_start[0] + t_j * (color_end[0] - color_start[0]),
                            color_start[1] + t_j * (color_end[1] - color_start[1]),
                            color_start[2] + t_j * (color_end[2] - color_start[2]),
                        )

                        arcade.draw_line(pos_i[0], pos_i[1], pos_j[0], pos_j[1], arcade.color.BLACK, 2)
                        arcade.draw_circle_filled(pos_i[0], pos_i[1], node_size_i, node_color_i)
                        arcade.draw_circle_filled(pos_j[0], pos_j[1], node_size_j, node_color_j) 
                        arcade.draw_circle_filled(pos_i[0], pos_i[1], node_size_i * perimeter_i * 0.5, arcade.color.GOLD)
                        arcade.draw_circle_filled(pos_j[0], pos_j[1], node_size_j * perimeter_j * 0.5, arcade.color.GOLD) 

                        # node for frontier barycenter
                        pos_x_i = self.features[i]["frontier_barycenter_x"]
                        pos_y_i = self.features[i]["frontier_barycenter_y"]
                        pos_i = Vector2D(pos_x_i, pos_y_i)
                        pos_i = self.map.grid_to_world(pos_i)
                        pos_i = pos_i.array + np.array(self.drone.size_area) / 2
                        
                        pos_x_j = self.features[j]["frontier_barycenter_x"]
                        pos_y_j = self.features[j]["frontier_barycenter_y"]
                        pos_j = Vector2D(pos_x_j, pos_y_j)
                        pos_j = self.map.grid_to_world(pos_j)
                        pos_j = pos_j.array + np.array(self.drone.size_area) / 2

                        node_size_i = (self.features[i]["frontier_size"] / 10)
                        node_size_j = (self.features[j]["frontier_size"] / 10)

                        arcade.draw_circle_filled(pos_i[0], pos_i[1], node_size_i, arcade.color.BLUE)
                        arcade.draw_circle_filled(pos_j[0], pos_j[1], node_size_j, arcade.color.BLUE)

        
    def update(self):
        drone_pos = self.map.world_to_grid(self.drone.drone_position)
        self.labels_map = np.zeros((self.map.get_width(), self.map.get_height()), dtype=int)    
        wait = {(drone_pos.x, drone_pos.y)}
        current_label = 1
        total_cell_tiles = []
        while len(wait) > 0:
            i, j = wait.pop()
            if self.labels_map[i, j] != 0:
                continue
            new_wait, cell_tiles = self.propagate_cell(i, j, current_label)
            current_label += 1
            total_cell_tiles.append(cell_tiles)
            wait = wait | new_wait
        self.total_cell_tiles = total_cell_tiles

        features, A = self.get_features()
        self.graph = nx.from_numpy_array(A)
        drone_node = self.labels_map[drone_pos.x, drone_pos.y]

        if len(features) > 0:
            self.add_drone_distance(A, features, drone_node)

        self.features = features
        self.adjacency_matrix = A


