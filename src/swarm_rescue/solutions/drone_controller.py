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
    going_to_return_zone = State('Going to return zone')
    stay_in_return_zone = State('Stay in return zone')

    force_drone_stuck = approaching_wounded.to(roaming)

    # transitions
    cycle = (
        roaming.to(approaching_wounded, cond="wounded_visible", on="before_approaching_wounded") |
        roaming.to(going_to_wounded, cond="found_wounded", on="before_going_to_wounded") |
        roaming.to(going_to_center, cond="no_gps_time_limit") |
        roaming.to(going_to_return_zone, cond="is_drone_finished", on="before_going_to_return_zone") |
        roaming.to(roaming) |

        going_to_wounded.to(approaching_wounded, cond="wounded_visible", on="before_approaching_wounded") |
        going_to_wounded.to(roaming, cond="lost_wounded_found") |
        going_to_wounded.to(going_to_wounded) |

        approaching_wounded.to(roaming, cond="no_wounded") |
        approaching_wounded.to(going_to_center, cond="grasped_wounded", on="before_going_to_center") |
        approaching_wounded.to(approaching_wounded) |

        going_to_center.to(approaching_center, cond="found_center", on='before_approaching_center') |
        going_to_center.to(roaming, cond="lost_wounded", ) |
        going_to_center.to(going_to_center) |

        approaching_center.to(going_to_center, cond="lost_center") |
        approaching_center.to(roaming, cond="lost_wounded") |
        approaching_center.to(approaching_center) |

        going_to_return_zone.to(stay_in_return_zone, cond="arrived_to_return_zone", on="before_stay_in_return_zone") |
        going_to_return_zone.to(going_to_return_zone) |

        stay_in_return_zone.to(going_to_return_zone, cond="out_of_return_zone", on="before_going_to_return_zone") |
        stay_in_return_zone.to(stay_in_return_zone)
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
        if not self.drone.found_wounded:
            return False

        path = self.drone.get_path(self.drone.wounded_target)
        if path is None:
            return False
        return True
    
    def wounded_visible(self) -> bool:
        return self.drone.wounded_visible and self.drone.found_wounded
    
    def lost_wounded_found(self) -> bool:
        if not self.drone.found_wounded:
            return True

        path = self.drone.get_path(self.drone.wounded_target)
        if path is None:
            return True
        return False

    def no_wounded(self) -> bool:
        return not self.drone.wounded_visible and not self.drone.base.grasper.grasped_entities
    
    def grasped_wounded(self) -> bool:
        return len(self.drone.base.grasper.grasped_entities) > 0
    
    def found_center(self) -> bool:
        return self.drone.rescue_center_dist is not None
    
    def lost_center(self) -> bool:
        return self.drone.rescue_center_dist is None
    
    def lost_wounded(self) -> bool:
        return not self.drone.base.grasper.grasped_entities
    
    def is_drone_finished(self) -> bool:
        return self.drone.is_simulation_time_limit_reached()
    
    def arrived_to_return_zone(self) -> bool:
        return self.drone.is_inside_return_area
    
    def out_of_return_zone(self) -> bool:
        return not self.drone.is_inside_return_area
    
    
    ## actions

    def before_cycle(self, event: str, source: State, target: State, message: str = "") -> None:
        if not self.debug_mode: return
        message = ". " + message if message else ""
        print(f"Running {event} from {source.id} to {target.id}{message}")

    @roaming.enter
    def on_enter_roaming(self) -> None:
        pass


    def before_going_to_wounded(self) -> None:
        self.drone.reset_path()

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self) -> None:
        if len(self.drone.path)==0 and self.drone.next_waypoint is None:
            path = self.drone.get_path(self.drone.wounded_target)
            self.drone.set_path(path)
        self.command = self.drone.localizer.get_control_from_path()

    def before_approaching_wounded(self) -> None:
        self.drone.reset_path()
    
    @approaching_wounded.enter
    def on_enter_approaching_wounded(self) -> None:
        self.command = self.drone.localizer.get_control_from_semantic()
        if self.drone.wounded_target is None:
            self.command["grasper"] = 0
            return
        dist = self.drone.wounded_target.distance(self.drone.drone_position)
        if dist < 80:
            self.command["grasper"] = 1
        else:
            self.command["grasper"] = 0

    def before_going_to_center(self) -> None:
        self.drone.drone_angle_offset = np.pi
        path = self.drone.get_path(self.drone.rescue_center_position)
        self.drone.set_path(path)

    @going_to_center.enter
    def on_enter_going_to_center(self) -> None:
        self.drone.drone_angle_offset = np.pi
        if self.drone.target is None:
            path = self.drone.get_path(self.drone.rescue_center_position)
            self.drone.set_path(path)
        self.command = self.drone.localizer.get_control_from_path()
        self.command["grasper"] = 1

    @going_to_center.exit
    def on_exit_going_to_center(self) -> None:
        self.drone.drone_angle_offset = 0

    def before_approaching_center(self) -> None:
        self.drone.reset_path()
        self.drone.localizer.time_approaching_center = 0
    
    @approaching_center.enter
    def enter_approaching_center(self) -> None:
        self.command = self.drone.localizer.get_control_from_semantic()
        self.command["grasper"] = 1
        self.drone.localizer.time_approaching_center += 1

    def before_going_to_return_zone(self) -> None:
        self.drone.drone_angle_offset = np.pi
        path = self.drone.get_path(self.drone.return_zone_position)
        self.drone.set_path(path)

    @going_to_return_zone.enter
    def on_enter_going_to_return_zone(self) -> None:
        self.drone.drone_angle_offset = np.pi
        if self.drone.target is None:
            path = self.drone.get_path(self.drone.return_zone_position)
            self.drone.set_path(path)
        self.command = self.drone.localizer.get_control_from_path()
        self.command["grasper"] = 0

    def before_stay_in_return_zone(self) -> None:
        self.command["forward"] = 0
        self.command["lateral"] = 0
        self.command["rotation"] = 0
        self.command["grasper"] = 0

    @stay_in_return_zone.enter
    def on_enter_stay_in_return_zone(self) -> None:
        self.drone.reset_path()


