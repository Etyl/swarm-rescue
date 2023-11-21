from statemachine import StateMachine, State
from solutions.mapper.mapper import Map
from spg_overlay.entities.drone_abstract import DroneAbstract
import numpy as np
from enum import Enum

class Zone(Enum):
    EMPTY = 0
    OBSTACLE = 1
    WOUNDED = 2
    RESCUE_CENTER = 3
    INEXPLORED = -1

class RoamerController(StateMachine):
    start_roaming = State('Start Roaming', initial=True)
    searching_for_target = State('Searching for target')
    going_to_target = State('Going to target')

    cycle = (
        start_roaming.to(searching_for_target) |
        searching_for_target.to(going_to_target) |
        going_to_target.to(searching_for_target)
    )

    def __init__(self, drone: DroneAbstract, map: Map, debug_mode: bool = False):
        self.drone = drone
        self.map = map
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.debug_mode = debug_mode

        super(RoamerController, self).__init__()

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        message = ". " + message if message else ""
        return f"Running {event} from {source.id} to {target.id}{message}"

    def on_enter_start_roaming(self):
        print("Entering start roaming state")

    def on_exit_red(self):
        print("Go ahead!")

class Roamer:
    def __init__(self, drone: DroneAbstract, map: Map, current_drone_pos: np.ndarray):
        self.current_drone_pos = current_drone_pos
        self.drone = drone
        self.map = map
        self.controller = RoamerController(drone, map, True)
    
    def find_next_unexeplored_target():
        """
        Find the closest unexplored target from the drone's curretn position
        It comes to finding the closest INEXPLORED point which is next to a explored point in the map
        """
        # convert the current drone world position to grid position

        # find the closest unexplored target

        # return the target

        rows, cols = map.shape
        current_row, current_col = self.current_drone_pos

        def is_valid_move(row, col):
            return 0 <= row < rows and 0 <= col < cols

        # Define the possible moves (up, down, left, right)
        moves = [(-1, 0), (1, 0), (0, -1), (0, 1)]

        # Helper function to check if a cell has a neighbor with Zone value > 0
        def has_valid_neighbor(row, col):
            for dr, dc in moves:
                new_row, new_col = row + dr, col + dc
                if is_valid_move(new_row, new_col) and game_map.matrix[new_row, new_col].value > 0:
                    return True
            return False

        # BFS to find the closest unexplored point with a valid neighbor
        queue = [(current_row, current_col)]
        visited = set()

        while queue:
            current_row, current_col = queue.pop(0)
            if game_map.matrix[current_row, current_col] == Zone.INEXPLORED and has_valid_neighbor(current_row, current_col):
                return np.array([current_row, current_col])

            visited.add((current_row, current_col))

            for dr, dc in moves:
                new_row, new_col = current_row + dr, current_col + dc
                if is_valid_move(new_row, new_col) and (new_row, new_col) not in visited:
                    queue.append((new_row, new_col))

        # If no unexplored point with a valid neighbor is found
        return None

    
        