import heapq
from typing import Deque, Tuple, List, Set, Optional
import numpy as np
from collections import deque
import networkx as nx
import arcade
import torch
from torch_geometric.data import Data

from solutions.mapper.mapper import Map, Zone  # type: ignore
from solutions.utils.types import Vector2D, DroneData  # type: ignore
from solutions.mapper.gqn import GQN

FREE_TILE = 0
OBSTACLE_TILE = 1
UNKNOWN_TILE = 2
MAX_CELL_SIZE = 800
MAX_CELL_RADIUS = 35

class FeatureEnum:
    size = 12

    AREA=0
    PERIMETER=1
    BARYCENTER_X=2
    BARYCENTER_Y=3
    FRONTIER_SIZE=4
    FRONTIER_BARYCENTER_X=5
    FRONTIER_BARYCENTER_Y=6
    FRONTIER_SIZE_RATIO=7
    DRONE_COUNT=8
    TARGET_COUNT=9
    TARGET_PATH=10
    DRONE_DISTANCE=11

class InfoEnum:
    TIMESTEP = 0
    TARGET_DISTANCE = 1
    MAP_WIDTH = 2
    MAP_HEIGHT = 3
    MAX_CELL_SIZE = 4
    MAX_CELL_RADIUS = 5


class GraphMap:
    def __init__(self, map: Map, resolution: int, filename:str=None, gqn_file=None):
        self.map = map
        self.resolution = resolution
        self.map_width = map.get_width() // resolution
        self.map_height = map.get_height() // resolution
        self.labels_map : np.ndarray[int] = np.zeros((self.map_width, self.map_height), dtype=int)
        self.total_cell_tiles : List[List[Tuple[int, int]]] = []
        self.adjacency_matrix : Optional[np.ndarray[int]]  = None
        self.graph : Optional[nx.Graph] = None
        self.features : Optional[np.ndarray[float]] = None
        self.selected_node : Optional[int] = None
        self.filename = filename

        self.gqn = None
        self.device = "cpu"
        if gqn_file:
            self.gqn = GQN(12, 1, 16).to(self.device)
            gqn_dict = torch.load(gqn_file, weights_only=False, map_location=torch.device(self.device))
            self.gqn.load_state_dict(gqn_dict)

        self.best_nodes_ids = []
        self.best_nodes_probabilities = []

    def get_neighbours(self, i0: int, j0: int, include_unknown=False):
        return [
            (i, j)
            for i in range(i0 - 1, i0 + 2)
            for j in range(j0 - 1, j0 + 2)
            if (
                0 <= i < self.map_width 
                and 0 <= j < self.map_height
                and (i != i0 or j != j0)
                and (include_unknown or self.labels_map[i, j] == 0)
                and (
                    self.get_map_tile(i, j) == Zone.EMPTY
                    or (include_unknown and self.get_map_tile(i, j) == Zone.UNEXPLORED)
                )
            )
        ]
    
    def get_map_tile(self, i: int, j: int) -> int:
        return self.map.map[i * self.resolution, j * self.resolution]
    
    def world_to_grid(self, pos: Tuple[float,float]) -> Tuple[int,int]:
        """
        converts world coordinates to grid coordinates
        """
        y =  self.map_height - int(pos[1] / (self.map.resolution * self.resolution) + self.map_height / 2)
        x = int(pos[0] / (self.map.resolution * self.resolution) + self.map_width / 2)
        # crop the values to the map size
        x = max(0, min(x, self.map_width- 1))
        y = max(0, min(y, self.map_height - 1))
        return x, y
    
    def grid_to_world(self, pos: Tuple[int,int]) -> Tuple[float,float]:
        """
        converts grid coordinates to world coordinates
        """
        y = (self.map_height - pos[1]) * (self.map.resolution * self.resolution) - self.map_height * (self.map.resolution * self.resolution) / 2
        x = (pos[0] - self.map_width / 2) * (self.map.resolution * self.resolution)
        return x, y
    
    # TODO optimize
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

    # TODO optimize
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
                    if self.labels_map[i, j] != label and self.get_map_tile(i, j) == Zone.EMPTY:
                        A[label - 1, self.labels_map[i, j] - 1] = 1
                        A[self.labels_map[i, j] - 1, label - 1] = 1
                        perimeter += 1
                    if self.get_map_tile(i, j) == Zone.UNEXPLORED:
                        frontier.add((i0, j0))

            barycenter[0] /= len(self.total_cell_tiles[k])
            barycenter[1] /= len(self.total_cell_tiles[k])
            frontier_barycenter = [barycenter[0], barycenter[1]]
            if len(frontier) > 0:
                frontier_barycenter[0] = sum([p[0] for p in frontier]) / len(frontier)
                frontier_barycenter[1] = sum([p[1] for p in frontier]) / len(frontier)

            node_features = np.zeros(FeatureEnum.size)
            node_features[FeatureEnum.AREA] = len(self.total_cell_tiles[k])
            node_features[FeatureEnum.PERIMETER] = perimeter
            node_features[FeatureEnum.BARYCENTER_X] = barycenter[0]
            node_features[FeatureEnum.BARYCENTER_Y] = barycenter[1]
            node_features[FeatureEnum.FRONTIER_SIZE] = len(frontier)
            node_features[FeatureEnum.FRONTIER_BARYCENTER_X] = frontier_barycenter[0]
            node_features[FeatureEnum.FRONTIER_BARYCENTER_Y] = frontier_barycenter[1]

            if perimeter > 0:
                node_features[FeatureEnum.FRONTIER_SIZE_RATIO] = node_features[FeatureEnum.FRONTIER_SIZE] / node_features[FeatureEnum.PERIMETER]

            features.append(node_features)

        features = np.array(features, dtype=np.float32)
        return features, A

    def get_drone_features(self, features, drone_positions: List[Vector2D], drone_targets: List[Vector2D]):
        def node_distance_func(target):
            return lambda x: (x[FeatureEnum.BARYCENTER_X]-target[0])**2+(x[FeatureEnum.BARYCENTER_Y]-target[1])**2

        for drone_pos in drone_positions:
            drone_pos = self.world_to_grid(drone_pos.array)
            node_pos = np.argmin(list(map(node_distance_func(drone_pos), features)))
            features[node_pos][FeatureEnum.DRONE_COUNT] += 1

        for drone_target in drone_targets:
            if drone_target is not None:
                drone_target = self.world_to_grid(drone_target.array)
                node_target = np.argmin(list(map(node_distance_func(drone_target), features)))
                features[node_target][FeatureEnum.TARGET_COUNT] += 1


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
                        features[current_node][FeatureEnum.BARYCENTER_X],
                        features[current_node][FeatureEnum.BARYCENTER_Y],
                    )
                    neighbor_pos = (
                        features[neighbor][FeatureEnum.BARYCENTER_X],
                        features[neighbor][FeatureEnum.BARYCENTER_Y],
                    )
                    new_dist = current_dist + np.sqrt(
                        (current_node_pos[0] - neighbor_pos[0]) ** 2
                        + (current_node_pos[1] - neighbor_pos[1]) ** 2
                    )
                    if new_dist < dist[neighbor]:
                        dist[neighbor] = new_dist
                        features[neighbor][FeatureEnum.TARGET_PATH] = features[current_node][FeatureEnum.TARGET_PATH]+features[neighbor][FeatureEnum.DRONE_COUNT]
                        heapq.heappush(priority_queue, (new_dist, neighbor))

        for k in range(num_nodes):
            features[k][FeatureEnum.DRONE_DISTANCE] = dist[k]

    def get_grid_barycenter(self, x, drone, frontier=False):
        pos_x = x[FeatureEnum.FRONTIER_BARYCENTER_X if frontier else FeatureEnum.BARYCENTER_X]
        pos_y = x[FeatureEnum.FRONTIER_BARYCENTER_Y if frontier else FeatureEnum.BARYCENTER_Y]
        pos = (pos_x, pos_y)
        pos = self.grid_to_world(pos)
        pos = np.array(pos) + np.array(drone.size_area) / 2
        return pos

    def draw(self, drone):
        if self.graph is None:
            return

        selected_feature = FeatureEnum.FRONTIER_SIZE
        max_val = np.max(list(map(
            lambda x: x[selected_feature],
            self.features
        )))

        for i in range(len(self.adjacency_matrix)):
            for j in range(i + 1, len(self.adjacency_matrix)):

                if self.adjacency_matrix[i][j] == 1:

                    pos_i = self.get_grid_barycenter(self.features[i], drone)
                    pos_j = self.get_grid_barycenter(self.features[j], drone)

                    node_size_i = (self.features[i][FeatureEnum.AREA] / MAX_CELL_SIZE + 1)*5
                    node_size_j = (self.features[j][FeatureEnum.AREA] / MAX_CELL_SIZE + 1)*5

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

                    node_size_i = (self.features[i][FeatureEnum.FRONTIER_SIZE] / 10)
                    node_size_j = (self.features[j][FeatureEnum.FRONTIER_SIZE] / 10)

                    #arcade.draw_circle_filled(pos_i[0], pos_i[1], node_size_i, arcade.color.BLUE)
                    #arcade.draw_circle_filled(pos_j[0], pos_j[1], node_size_j, arcade.color.BLUE)

                    if self.selected_node is not None:
                        pos = self.get_grid_barycenter(self.features[self.selected_node], drone)
                        arcade.draw_circle_outline(pos[0], pos[1], 10, arcade.color.GREEN, 2)

        for id, prob in zip(self.best_nodes_ids, self.best_nodes_probabilities):
            pos = self.get_grid_barycenter(self.features[id], drone)
            color_start = arcade.color.WHITE
            color_end = arcade.color.GREEN
            t = min(prob, 1)
            node_color = (
                color_start[0] + t * (color_end[0] - color_start[0]),
                color_start[1] + t * (color_end[1] - color_start[1]),
                color_start[2] + t * (color_end[2] - color_start[2]),
            )
            arcade.draw_circle_filled(pos[0], pos[1], 10, node_color)

    def save_data(self, filename, reward, info):
        if self.features is None:
            return

        with open(filename, "a") as f:
            # Write the graph
            f.write(f"{len(self.features)}\n")
            edges = self.graph.edges()
            f.write(f"{len(edges)}\n")
            for edge in edges:
                f.write(f"{edge[0]} {edge[1]}\n")
            # Write the features
            for i, x in enumerate(self.features):
                f.write(f"{i} {' '.join(list(map(str,x)))}\n")
            # action
            f.write(f"{self.selected_node}\n")
            # rewards
            f.write(f"{' '.join(list(map(str,reward)))}\n")
            # general info
            f.write(f"{' '.join(list(map(str,info)))}\n")


    def update(self, drone) -> None:
        drone_pos = self.world_to_grid(drone.drone_position.array)
        self.labels_map = np.zeros((self.map_width, self.map_height), dtype=int)
        wait = {(drone_pos[0], drone_pos[1])}
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
        drone_node = self.labels_map[drone_pos[0], drone_pos[1]]-1

        if len(features) > 0:
            drone_positions = [data[1] for data in drone.drone_positions.values()]
            drone_targets = [data.target for data in drone.drone_list]
            self.get_drone_features(features, drone_positions, drone_targets)
            self.add_drone_distance(A, features, drone_node)

        self.features = features
        self.adjacency_matrix = A

    def get_normalized_features(self, drone) -> torch.Tensor:
        info = drone.get_path_info()
        features = torch.tensor(self.features)
        features[:, FeatureEnum.AREA] = features[:, FeatureEnum.AREA] / info[InfoEnum.MAX_CELL_SIZE]
        features[:, FeatureEnum.PERIMETER] = features[:, FeatureEnum.AREA] / (4 * info[InfoEnum.MAX_CELL_RADIUS])
        features[:, FeatureEnum.BARYCENTER_X] = features[:, FeatureEnum.BARYCENTER_X] / info[InfoEnum.MAP_HEIGHT]
        features[:, FeatureEnum.BARYCENTER_Y] = features[:, FeatureEnum.BARYCENTER_Y] / info[InfoEnum.MAP_WIDTH]
        features[:, FeatureEnum.FRONTIER_BARYCENTER_X] = features[:, FeatureEnum.FRONTIER_BARYCENTER_X] / info[
            InfoEnum.MAP_HEIGHT]
        features[:, FeatureEnum.FRONTIER_BARYCENTER_Y] = features[:, FeatureEnum.FRONTIER_BARYCENTER_Y] / info[
            InfoEnum.MAP_WIDTH]
        features[:, FeatureEnum.DRONE_DISTANCE] = torch.clip(
            features[:, FeatureEnum.DRONE_DISTANCE] / (info[InfoEnum.MAP_WIDTH] + info[InfoEnum.MAP_HEIGHT]), 0, 1)
        features[:, FeatureEnum.FRONTIER_SIZE] = features[:, FeatureEnum.FRONTIER_SIZE] / (
            2 * info[InfoEnum.MAX_CELL_RADIUS])
        return features

    def get_node_center(self, selected_node_id:int, drone) -> Vector2D:
        frontier_center = (
            self.features[selected_node_id][FeatureEnum.FRONTIER_BARYCENTER_X],
            self.features[selected_node_id][FeatureEnum.FRONTIER_BARYCENTER_Y]
        )
        frontier_center = self.grid_to_world(frontier_center)
        frontier_center = Vector2D(pointList=frontier_center)
        self.selected_node = selected_node_id

        if self.filename is not None:
            self.save_data(self.filename, drone.get_reward(), drone.get_path_info())

        drone.last_frontier_target = frontier_center

        return frontier_center

    def get_best_node(self, drone) -> Optional[Vector2D]:
        if self.gqn is None:
            return self.get_naive_node(drone)

        if len(self.features) == 0 or self.graph is None:
            return None

        features = self.get_normalized_features(drone)
        graph = Data(x=features, edge_index=torch.tensor(list(self.graph.edges())).t().long())
        info = drone.get_path_info()
        exploration = drone.get_reward()[1] * info[InfoEnum.MAP_HEIGHT] * info[InfoEnum.MAP_WIDTH] / 1000
        exploration = torch.tensor(exploration)
        q_values = self.gqn.forward_single(graph, exploration).detach().cpu().numpy()

        frontier_mask = self.features[:,FeatureEnum.FRONTIER_SIZE] <= 1
        q_values[frontier_mask] = 0
        selected_node_id = np.argmax(q_values)

        self.best_nodes_ids = np.arange(0,len(self.features))[~frontier_mask]
        probs = q_values[~frontier_mask]
        if len(probs)>0:
            probs = probs - np.min(probs) + 0.0001
            probs = probs / np.max(probs)
        self.best_nodes_probabilities = probs

        return self.get_node_center(selected_node_id, drone)
    
    def get_naive_node(self, drone) -> Optional[Vector2D]:
        if len(self.features) == 0:
            return None

        distances = np.array([
            self.features[k][FeatureEnum.DRONE_DISTANCE] / np.sqrt((self.map_width**2 + self.map_height**2)) if self.features[k][FeatureEnum.FRONTIER_SIZE] > 1 else np.inf
            for k in range(len(self.features))
        ])
        
        temperature = 0.1
        valid_mask = distances != np.inf
        valid_distances = distances[valid_mask]
        
        if len(valid_distances) == 0:
            return None
        
        exp_values = np.exp(-valid_distances / temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        valid_indices = np.where(valid_mask)[0]
        selected_node_id = np.random.choice(valid_indices, p=probabilities)

        self.best_nodes_ids = valid_indices
        self.best_nodes_probabilities = probabilities

        frontier_center = (
            self.features[selected_node_id][FeatureEnum.FRONTIER_BARYCENTER_X],
            self.features[selected_node_id][FeatureEnum.FRONTIER_BARYCENTER_Y]
        )
        frontier_center = self.grid_to_world(frontier_center)
        frontier_center = Vector2D(pointList=frontier_center)
        self.selected_node = selected_node_id

        if self.filename is not None:
            self.save_data(self.filename, drone.get_reward(), drone.get_path_info())

        drone.last_frontier_target = frontier_center

        return frontier_center



