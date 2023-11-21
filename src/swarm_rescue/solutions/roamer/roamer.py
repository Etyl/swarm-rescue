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

    
        