import heapq
from typing import Deque, Tuple, List, Set, Optional
import numpy as np
from collections import deque, defaultdict
import networkx as nx
import arcade

from solutions.mapper.mapper import Map, Zone  # type: ignore
from solutions.utils.types import Vector2D, DroneData  # type: ignore

FREE_TILE = 0
OBSTACLE_TILE = 1
UNKNOWN_TILE = 2
MAX_CELL_SIZE = 800
MAX_CELL_RADIUS = 35


class GraphMap:
    def __init__(self, map: Map):
        self.map = map
        self.labels_map : np.ndarray[int] = np.zeros((map.get_width(), map.get_height()), dtype=int)
        self.total_cell_tiles : List[List[Tuple[int, int]]] = []
        self.adjacency_matrix : Optional[np.ndarray]  = None
        self.graph : Optional[nx.Graph] = None
        self.features : Optional[np.ndarray] = None

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
                    self.map.map[i,j] == Zone.EMPTY
                    or (include_unknown and self.map.map[i,j] == Zone.UNEXPLORED)
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
            neighbors = self.get_neighbours(i, j)
            for neigh in neighbors:
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
                neighbors = self.get_neighbours(i0, j0, include_unknown=True)
                for i, j in neighbors:
                    if self.labels_map[i, j] != label and self.map.map[i,j] == Zone.EMPTY:
                        A[label - 1, self.labels_map[i, j] - 1] = 1
                        A[self.labels_map[i, j] - 1, label - 1] = 1
                        perimeter += 1
                    if self.map.map[i,j] == Zone.UNEXPLORED:
                        frontier.add((i0, j0))

            barycenter[0] /= len(self.total_cell_tiles[k])
            barycenter[1] /= len(self.total_cell_tiles[k])
            frontier_barycenter = [barycenter[0], barycenter[1]]
            if len(frontier) > 0:
                frontier_barycenter[0] = sum([p[0] for p in frontier]) / len(frontier)
                frontier_barycenter[1] = sum([p[1] for p in frontier]) / len(frontier)

            node_features = defaultdict(float)
            node_features["area"] = len(self.total_cell_tiles[k])
            node_features["perimeter"] = perimeter
            node_features["barycenter_x"] = barycenter[0]
            node_features["barycenter_y"] = barycenter[1]
            node_features["frontier_size"] = len(frontier)
            node_features["frontier_barycenter_x"] = frontier_barycenter[0]
            node_features["frontier_barycenter_y"] = frontier_barycenter[1]

            if perimeter > 0:
                node_features["frontier_size_ratio"] = node_features["frontier_size"] / node_features["perimeter"]

            features.append(node_features)

        return features, A

    def get_drone_features(self, features, drone_positions: List[Vector2D], drone_targets: List[Vector2D]):
        def node_distance_func(target):
            return lambda x: Vector2D(x["barycenter_x"], x["barycenter_y"]).distance(target)

        for drone_pos in drone_positions:
            drone_pos = self.map.world_to_grid(drone_pos)
            node_pos = np.argmin(list(map(node_distance_func(drone_pos),features)))
            features[node_pos]["drone_count"] += 1

        for drone_target in drone_targets:
            if drone_target is not None:
                drone_target = self.map.world_to_grid(drone_target)
                node_target = np.argmin(list(map(node_distance_func(drone_target),features)))
                features[node_target]["target_count"] += 1


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
                if A[current_node, neighbor] > 0 and not visited[neighbor]:
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
                        features[neighbor]["target_path"] = features[current_node]["target_path"]+features[neighbor]["drone_count"]
                        heapq.heappush(priority_queue, (new_dist, neighbor))

        for k in range(num_nodes):
            features[k]["drone_distance"] = dist[k]

    def get_grid_barycenter(self, x, drone, frontier=False):
        pos_x = x[f"{'frontier_' if frontier else ''}barycenter_x"]
        pos_y = x[f"{'frontier_' if frontier else ''}barycenter_y"]
        pos = Vector2D(pos_x, pos_y)
        pos = self.map.grid_to_world(pos)
        pos = pos.array + np.array(drone.size_area) / 2
        return pos

    def draw(self, drone):
        if self.graph is None:
            return

        selected_feature = "target_path"
        max_val = np.max(list(map(
            lambda x: x[selected_feature],
            self.features
        )))

        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):

                if self.adjacency_matrix[i][j] == 1:

                    pos_i = self.get_grid_barycenter(self.features[i], drone)
                    pos_j = self.get_grid_barycenter(self.features[j], drone)

                    node_size_i = (self.features[i]["area"] / MAX_CELL_SIZE + 1)*5
                    node_size_j = (self.features[j]["area"] / MAX_CELL_SIZE + 1)*5

                    # Gradient of color depending on the frontier size
                    color_start = arcade.color.BLACK
                    color_end = arcade.color.RED
                    t_i = min(self.features[i][selected_feature] / max_val, 1)
                    t_j = min(self.features[j][selected_feature] / max_val, 1)
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
                    # node for frontier barycenter
                    pos_i = self.get_grid_barycenter(self.features[i], drone, frontier=True)
                    pos_j = self.get_grid_barycenter(self.features[i], drone, frontier=True)

                    node_size_i = (self.features[i]["frontier_size"] / 10)
                    node_size_j = (self.features[j]["frontier_size"] / 10)

                    arcade.draw_circle_filled(pos_i[0], pos_i[1], node_size_i, arcade.color.BLUE)
                    arcade.draw_circle_filled(pos_j[0], pos_j[1], node_size_j, arcade.color.BLUE)



    def update(self, drone):
        drone_pos = self.map.world_to_grid(drone.drone_position)
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
            drone_positions = [data[1] for data in drone.drone_positions.values()]
            drone_targets = [data.target for data in drone.drone_list]
            self.get_drone_features(features, drone_positions, drone_targets)
            self.add_drone_distance(A, features, drone_node)

        self.features = features
        self.adjacency_matrix = A


