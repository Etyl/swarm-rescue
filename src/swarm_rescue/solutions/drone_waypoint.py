"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
import numpy as np
from typing import Optional
from enum import Enum
from statemachine import StateMachine, State


from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean, normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor


class DroneController(StateMachine):

    # states  
    roaming = State('Roaming')
    going_to_wounded = State('Going to wounded', initial=True)
    approaching_wounded = State('Approaching wounded')
    going_to_center = State('Going to center')
    approaching_center = State('Approaching to center')

    # transitions
    cycle = (
        roaming.to(approaching_wounded, cond="found_wounded") |
        going_to_wounded.to(approaching_wounded, cond="found_wounded") |

        going_to_wounded.to(roaming, cond="lost_route") |

        # if wounded captured by someone else
        approaching_wounded.to(roaming, cond="no_wounded") |

        approaching_wounded.to(going_to_center, cond="grasped_wounded") |

        going_to_center.to(approaching_center, cond="found_center") |

        approaching_center.to(roaming, cond="lost_wounded") |
        going_to_center.to(roaming, cond="lost_wounded")
    )


    
    def __init__(self, drone : DroneAbstract, debug_mode: bool = False):
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
        return len(self.base.grasper.grasped_entities) > 0
    
    def found_center(self):
        return self.drone.found_center
    
    def lost_wounded(self):
        return not self.drone.base.grasper.grasped_entities
    
    
    ## actions

    def before_cycle(self, event: str, source: State, target: State, message: str = ""):
        if not self.debug_mode: return
        message = ". " + message if message else ""
        return f"Running {event} from {source.id} to {target.id}{message}"

    @roaming.enter
    def on_enter_initial(self):
        # TODO: implement exploration
        self.drone.onRoute = False

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self):
        self.drone.path = self.drone.get_path(self.drone.get_position(), 0)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_wounded.to.itself()
    def loop_going_to_wounded(self):
        self.command = self.drone.get_control_from_path(self.drone.get_position())

    @approaching_wounded.enter
    def on_enter_approaching_wounded(self):
        self.drone.onRoute = False
    
    @approaching_wounded.to.itself()
    def loop_approaching_wounded(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1

    @going_to_center.enter
    def on_enter_going_to_center(self):
        self.drone.path = self.drone.get_path(self.drone.drone_position, 1)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_center.to.itself()
    def loop_going_to_center(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)
        self.command["grasper"] = 1

    @approaching_center.enter
    def on_enter_approaching_center(self):
        self.drone.onRoute = False
    
    @approaching_center.to.itself()
    def loop_approaching_center(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1


class DroneWaypoint(DroneAbstract):
    class Activity(Enum):
        """
        All the states of the drone as a state machine
        """
        ROAMING = 1
        GOING_TO_WONDED = 2
        GRASPING_WOUNDED = 3
        RETURNING_TO_CENTER = 4
        DROPPING_TO_CENTER = 5

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

        self.state = self.Activity.GOING_TO_WONDED
        self.onRoute = False # True if the drone is on the route to the waypoint
        self.path = []
        self.nextWaypoint = None # The next waypoint to go to
        self.controller = DroneController(self)

        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")


    def adapt_angle_direction(self, pos: list):
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.measured_compass_angle() != None:
            
            def angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

            pos = np.array(pos)
            waypoint = np.array(self.nextWaypoint)
            waypoint_vect = waypoint - pos
            ref_vect = np.array([1, 0])

            
            drone_angle = normalize_angle(self.measured_compass_angle())
            waypoint_angle = normalize_angle(angle(ref_vect, waypoint_vect))

            angle = normalize_angle(waypoint_angle - drone_angle)

            if angle > 0.5:
                return 1.0
            if angle < -0.5:
                return -1.0
            if angle > 0:
                angle
            return angle
        return 0
    
    def check_waypoint(self, pos):
        """
        checks if the drone has reached the waypoint
        """

        if np.linalg.norm(pos - self.nextWaypoint) < 20:
            return True
        return False
    
    # TODO: implement communication
    def define_message_for_all(self):
        pass
    
    # TODO: determine the position in no GPS zone
    def get_position(self):
        """
        returns the position of the drone
        """
        return self.measured_gps_position()

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """
        
        command = {"forward": 0.5,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        best_angle = 0

        found_wounded = False
        if (self.state is self.Activity.ROAMING
            or self.state is self.Activity.GRASPING_WOUNDED
            or self.state is self.Activity.GOING_TO_WONDED) \
                and detection_semantic is not None:
            scores = []
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    found_wounded = True
                    v = (data.angle * data.angle) + \
                        (data.distance * data.distance / 10 ** 5)
                    scores.append((v, data.angle, data.distance))

            # Select the best one among wounded persons detected
            best_score = 10000
            for score in scores:
                if score[0] < best_score:
                    best_score = score[0]
                    best_angle = score[1]

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (self.state is self.Activity.RETURNING_TO_CENTER and detection_semantic):
            for data in detection_semantic:
                if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER:
                    found_rescue_center = True
                    angles_list.append(data.angle)
                    is_near = (data.distance < 50)

            if found_rescue_center:
                best_angle = circular_mean(np.array(angles_list))

        if found_rescue_center or found_wounded:
            # simple P controller
            # The robot will turn until best_angle is 0
            kp = 2.0
            a = kp * best_angle
            a = min(a, 1.0)
            a = max(a, -1.0)
            command["rotation"] = a * angular_vel_controller_max

            # reduce speed if we need to turn a lot
            if abs(a) == 1:
                command["forward"] = 0.2

        if found_rescue_center and is_near:
            command["forward"] = 0
            command["rotation"] = random.uniform(0.5, 1)

        return found_wounded, found_rescue_center, command

    # TODO: implement pathfinder with map
    def get_path(self, pos, destination):
        """
        returns the path to the destination
        """

        if destination == 0:
            return [[-320,-180],[-200,150],[20, -200]]
        return [[250,150],[20, -200],[-200,150]]

    def get_control_from_path(self, pos):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        command["forward"] = 1.0
        command["rotation"] = self.adapt_angle_direction(pos)
        command["lateral"] = math.sin(command["rotation"])
        if self.check_waypoint(pos):
            if len(self.path) > 0:
                self.nextWaypoint = self.path.pop()
            else:
                self.nextWaypoint = None
                self.onRoute = False
        return command

    def control(self):

        found_wounded, found_center, command_semantic = self.process_semantic_sensor()

        print("=====================================")
        print(self.semantic_values())
        print(self.state, found_wounded, self.nextWaypoint, self.onRoute, self.base.grasper.grasped_entities, self.base.grasper.can_grasp)

        if self.state is self.Activity.ROAMING or self.Activity.GOING_TO_WONDED and found_wounded:
            self.state = self.Activity.GRASPING_WOUNDED

        elif self.state is self.Activity.GOING_TO_WONDED and self.onRoute and self.nextWaypoint is None:
            self.state = self.Activity.ROAMING

        # case if wounded captured by someone else
        elif self.state is self.Activity.GRASPING_WOUNDED and (not found_wounded and len(self.base.grasper.grasped_entities) == 0):
            self.state = self.Activity.ROAMING

        elif self.state is self.Activity.GRASPING_WOUNDED and len(self.base.grasper.grasped_entities) > 0:
            self.state = self.Activity.RETURNING_TO_CENTER

        elif self.state is self.Activity.RETURNING_TO_CENTER and found_center:
            self.state = self.Activity.DROPPING_TO_CENTER

        elif (self.state is (self.Activity.RETURNING_TO_CENTER or self.Activity.DROPPING_TO_CENTER)) and not self.base.grasper.grasped_entities:
            self.state = self.Activity.ROAMING

        print(self.state, found_wounded, self.nextWaypoint, self.onRoute, self.base.grasper.grasped_entities, self.base.grasper.can_grasp)

        drone_position = self.get_position()


        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        # Controls for the differents states

        if self.state is self.Activity.ROAMING:
            # TODO: implement exploration
            self.onRoute = False

        elif self.state is self.Activity.GOING_TO_WONDED:
            if not self.onRoute:
                self.path = self.get_path(drone_position, 0)
                self.nextWaypoint = self.path.pop()
                self.onRoute = True
            else:
                command = self.get_control_from_path(drone_position)
            
        
        elif self.state is self.Activity.RETURNING_TO_CENTER:
            if not self.onRoute:
                self.path = self.get_path(drone_position, 1)
                self.nextWaypoint = self.path.pop()
                self.onRoute = True
            command = self.get_control_from_path(drone_position)
            command["grasper"] = 1

        elif self.state is self.Activity.GRASPING_WOUNDED or self.state is self.Activity.DROPPING_TO_CENTER:
            self.onRoute = False
            command = command_semantic
            command["grasper"] = 1

            
            
        return command
