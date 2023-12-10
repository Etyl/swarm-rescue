from statemachine import StateMachine, State
import solutions



class DroneController(StateMachine):

    # states  
    roaming = State('Roaming', initial=True)
    going_to_wounded = State('Going to wounded')
    approaching_wounded = State('Approaching wounded')
    going_to_center = State('Going to center')
    approaching_center = State('Approaching to center')

    force_transition = roaming.to(going_to_wounded, on='before_going_to_wounded')

    # transitions
    cycle = (
        roaming.to(approaching_wounded, cond="found_wounded", on="before_approaching_wounded") |
        going_to_wounded.to(approaching_wounded, cond="found_wounded", on="before_approaching_wounded") |

        going_to_wounded.to(roaming, cond="lost_route") |

        # if wounded captured by someone else
        approaching_wounded.to(roaming, cond="no_wounded") |

        approaching_wounded.to(going_to_center, cond="grasped_wounded", on="before_going_to_center") |

        going_to_center.to(approaching_center, cond="found_center", on='before_approaching_center') |

        approaching_center.to(roaming, cond="lost_wounded") |
        going_to_center.to(roaming, cond="lost_wounded",) |

        roaming.to(roaming) |
        going_to_wounded.to(going_to_wounded) |
        approaching_wounded.to(approaching_wounded) |
        going_to_center.to(going_to_center) |
        approaching_center.to(approaching_center)
    )
 
    def __init__(self, drone : solutions.drone_waypoint.DroneWaypoint, debug_mode: bool = False):
        self.drone = drone
        self.command = {"forward": 0.0,
                        "lateral": 0.0,
                        "rotation": 0.0,
                        "grasper": 0}
        self.debug_mode = debug_mode

        super(DroneController, self).__init__()


    ## transitions conditions

    def found_wounded(self):
        return self.drone.found_wounded
    
    def lost_route(self):
        return self.drone.onRoute and self.drone.nextWaypoint is None
    
    def no_wounded(self):
        return not self.drone.found_wounded and not self.drone.base.grasper.grasped_entities
    
    def grasped_wounded(self):
        return len(self.drone.base.grasper.grasped_entities) > 0
    
    def found_center(self):
        return self.drone.found_center
    
    def lost_wounded(self):
        return not self.drone.base.grasper.grasped_entities
    
    
    ## actions

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        if not self.debug_mode: return
        message = ". " + message if message else ""
        print(f"Running {event} from {source.id} to {target.id}{message}")

    @roaming.enter
    def on_enter_roaming(self):
        # TODO: implement exploration
        # self.drone.onRoute = False
        self.drone.roaming = True

    def before_going_to_wounded(self):
        self.drone.path = self.drone.get_path(self.drone.rescue_center_position, 0)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)

    def before_approaching_wounded(self):
        self.drone.onRoute = False
        self.drone.roaming = False
    
    @approaching_wounded.enter
    def on_enter_approaching_wounded(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1

    def before_going_to_center(self):
        self.drone.path = self.drone.get_path(self.drone.rescue_center_position, 1)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_center.enter
    def on_enter_going_to_center(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)
        self.command["grasper"] = 1
        
        # ret = self.drone.keep_distance_from_walls()
        # if ret is not None:
        #     min_angle, min_dist = ret
        #     cos_angle = math.cos(min_angle)
        #     sin_angle = math.sin(min_angle)

        #     norm = max(abs(cos_angle),abs(sin_angle))
        #     cos_angle = cos_angle/norm
        #     sin_angle = sin_angle/norm 

        #     self.command["forward"] = cos_angle
        #     self.command["lateral"] = -sin_angle

    def before_approaching_center(self):
        self.drone.onRoute = False
    
    @approaching_center.enter
    def enter_approaching_center(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1

