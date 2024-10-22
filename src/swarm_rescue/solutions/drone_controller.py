from __future__ import annotations

from statemachine import StateMachine, State
import numpy as np
import typing

if typing.TYPE_CHECKING: # type: ignore
    from solutions.frontier_drone import FrontierDrone # type: ignore


class DroneController(StateMachine):

    # states  
    roaming = State('Roaming', initial=True)
    going_to_wounded = State('Going to wounded')
    approaching_wounded = State('Approaching wounded')
    going_to_center = State('Going to center')
    approaching_center = State('Approaching to center')

    force_drone_stuck = approaching_wounded.to(roaming)

    # transitions
    cycle = (
        roaming.to(approaching_wounded, cond="wounded_visible", on="before_approaching_wounded") |
        roaming.to(going_to_wounded, cond="found_wounded", on="before_going_to_wounded") |
        roaming.to(going_to_center, cond="no_gps_time_limit") |
        going_to_wounded.to(approaching_wounded, cond="wounded_visible", on="before_approaching_wounded") |

        going_to_wounded.to(roaming, cond="lost_wounded_found") |

        # if wounded captured by someone else
        approaching_wounded.to(roaming, cond="no_wounded") |

        approaching_wounded.to(going_to_center, cond="grasped_wounded", on="before_going_to_center") |

        going_to_center.to(approaching_center, cond="found_center", on='before_approaching_center') |

        approaching_center.to(going_to_center, cond="lost_center") |
        approaching_center.to(roaming, cond="lost_wounded") |
        going_to_center.to(roaming, cond="lost_wounded",) |

        roaming.to(roaming) |
        going_to_wounded.to(going_to_wounded) |
        approaching_wounded.to(approaching_wounded) |
        going_to_center.to(going_to_center) |
        approaching_center.to(approaching_center)
    )
 
    def __init__(self, drone : FrontierDrone, debug_mode: bool = False):
        self.drone : FrontierDrone = drone
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.debug_mode : bool = debug_mode

        super(DroneController, self).__init__()


    ## transitions conditions
    def no_gps_time_limit(self) -> bool:
        return self.drone.time_in_no_gps > 1000

    def found_wounded(self) -> bool:
        return self.drone.found_wounded
    
    def wounded_visible(self) -> bool:
        return self.drone.wounded_visible and self.drone.found_wounded
    
    def lost_wounded_found(self) -> bool:
        return not self.drone.found_wounded
    
    def no_wounded(self) -> bool:
        return not self.drone.wounded_visible and not self.drone.base.grasper.grasped_entities
    
    def grasped_wounded(self) -> bool:
        return len(self.drone.base.grasper.grasped_entities) > 0
    
    def found_center(self) -> bool:
        return self.drone.found_center and self.drone.is_near_center
    
    def lost_center(self) -> bool:
        return not self.drone.found_center
    
    def lost_wounded(self) -> bool:
        return not self.drone.base.grasper.grasped_entities
    
    
    ## actions

    def before_cycle(self, event: str, source: State, target: State, message: str = "") -> None:
        if not self.debug_mode: return
        message = ". " + message if message else ""
        print(f"Running {event} from {source.id} to {target.id}{message}")

    @roaming.enter
    def on_enter_roaming(self) -> None:
        # self.drone.onRoute = False
        pass


    def before_going_to_wounded(self) -> None:
        self.drone.onRoute = False
        self.drone.waypoint_index = None
        self.drone.path = []

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self) -> None:
        if len(self.drone.path)==0 and self.drone.waypoint_index is None:
            self.drone.path = self.drone.get_path(self.drone.wounded_target)
            if self.drone.path is None: 
                self.drone.path = []
                return
            else:
                self.drone.waypoint_index = 0
        self.command = self.drone.get_control_from_path(self.drone.drone_position)

    def before_approaching_wounded(self) -> None:
        self.drone.onRoute = False
        self.drone.waypoint_index = None
        self.drone.path = []
    
    @approaching_wounded.enter
    def on_enter_approaching_wounded(self) -> None:
        self.command = self.drone.get_control_from_semantic()
        if self.drone.wounded_target is None:
            self.command["grasper"] = 0
            return
        dist = self.drone.wounded_target.distance(self.drone.get_position())
        if dist < 80:
            self.command["grasper"] = 1
        else:
            self.command["grasper"] = 0

    def before_going_to_center(self) -> None:
        self.drone.drone_angle_offset = np.pi
        self.drone.path = self.drone.get_path(self.drone.rescue_center_position)
        if self.drone.path is None: 
            self.drone.path = []
            return
        self.drone.waypoint_index = 0
        self.drone.onRoute = True

    @going_to_center.enter
    def on_enter_going_to_center(self) -> None:
        self.drone.drone_angle_offset = np.pi
        if len(self.drone.path)==0 and self.drone.waypoint_index is None:
            self.drone.path = self.drone.get_path(self.drone.rescue_center_position)
            if self.drone.path is None: 
                self.drone.path = []
                return
            else:
                self.drone.waypoint_index = 0
        self.command = self.drone.get_control_from_path(self.drone.drone_position)
        self.command["grasper"] = 1

    @going_to_center.exit
    def on_exit_going_to_center(self) -> None:
        self.drone.drone_angle_offset = 0

    def before_approaching_center(self) -> None:
        self.drone.onRoute = False
        self.drone.waypoint_index = None
        self.drone.path = []
    
    @approaching_center.enter
    def enter_approaching_center(self) -> None:
        self.command = self.drone.get_control_from_semantic()
        self.command["grasper"] = 1


