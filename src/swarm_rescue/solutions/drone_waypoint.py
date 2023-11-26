"""
Simple random controller
The Drone will move forward and turn for a random angle when an obstacle is hit
"""
import math
import random
import numpy as np
from typing import Optional
from statemachine import StateMachine, State
import arcade
from collections import deque 

from spg_overlay.entities.drone_abstract import DroneAbstract
from spg_overlay.utils.misc_data import MiscData
from spg_overlay.utils.utils import circular_mean, normalize_angle
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from spg_overlay.utils.pose import Pose
from spg_overlay.entities.drone_distance_sensors import DroneSemanticSensor
from solutions.mapper.mapper import Map
from solutions.localizer.localizer import Localizer

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
        self.drone.onRoute = False

    def before_going_to_wounded(self):
        self.drone.path = self.drone.get_path(self.drone.drone_position, 0)
        self.drone.nextWaypoint = self.drone.path.pop()
        self.drone.onRoute = True

    @going_to_wounded.enter
    def on_enter_going_to_wounded(self):
        self.command = self.drone.get_control_from_path(self.drone.drone_position)

    def before_approaching_wounded(self):
        self.drone.onRoute = False
    
    @approaching_wounded.enter
    def on_enter_approaching_wounded(self):
        self.command = self.drone.command_semantic
        self.command["grasper"] = 1

    def before_going_to_center(self):
        self.drone.path = self.drone.get_path(self.drone.drone_position, 1)
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


class DroneWaypoint(DroneAbstract):

    def __init__(self,
                 identifier: Optional[int] = None,
                 misc_data: Optional[MiscData] = None,
                 **kwargs):
        super().__init__(identifier=identifier,
                         misc_data=misc_data,
                         display_lidar_graph=False,
                         **kwargs)

        self.onRoute = False # True if the drone is on the route to the waypoint
        self.path = []
        self.lastWaypoint = None # The last waypoint reached
        self.nextWaypoint = np.array([0,0]) # The next waypoint to go to
        self.drone_position = np.array([0,0]) # The position of the drone
        self.drone_angle = 0 # The angle of the drone
        self.found_wounded = False # True if the drone has found a wounded person
        self.found_center = False # True if the drone has found the rescue center
        self.command_semantic = None # The command to follow the wounded person or the rescue center
        self.controller = DroneController(self, debug_mode=False)
        self.last_angles = deque() # queue of the last angles
        self.angle_offset = np.pi/4 # The angle offset to go to the next waypoint

        self.wounded_found = []
        self.wounded_distance = 80 # The distance between wounded person to be considered as the same

        ## Debug controls

        self.debug_path = True # True if the path must be displayed
        self.debug_wounded = True
        self.debug_positions = True
        
        self.controller.force_transition()
        # to display the graph of the state machine (make sure to install graphviz, e.g. with "sudo apt install graphviz")
        # self.controller._graph().write_png("./graph.png")

        self.map = Map(area_world=self.size_area, resolution=16, lidar=self.lidar(), debug_mode=False)
        self.rescue_center_position = None
        
        self.localizer = Localizer()
        
        self.theorical_velocity = np.zeros(2)
        

    def adapt_angle_direction(self, pos: list):
        """
        gives the angle to turn to in order to go to the next waypoint
        """

        if self.drone_angle != None and self.nextWaypoint != None:
            
            def angle(v1, v2):
                return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])

            pos = np.array(pos)
            waypoint = np.array(self.nextWaypoint)
            waypoint_vect = waypoint - pos
            ref_vect = np.array([1, 0])

            
            drone_angle = normalize_angle(self.drone_angle)
            waypoint_angle = normalize_angle(angle(ref_vect, waypoint_vect))

            return normalize_angle(waypoint_angle - drone_angle)

        return 0
    
    
    def check_waypoint(self, pos):
        """
        checks if the drone has reached the waypoint
        """

        dist = np.linalg.norm(pos - self.nextWaypoint)
        if len(self.path) == 0: return dist < 20

        v1 = self.nextWaypoint - pos
        v2 = np.array(self.path[-1]) - np.array(self.nextWaypoint)
        turning_angle = np.dot(v1,v2)/(np.linalg.norm(v1)*np.linalg.norm(v2))

        # TODO: tune values
        return dist < 20 + (1+turning_angle)*20
        

    # TODO: implement communication
    def define_message_for_all(self):
        pass


    # TODO: improve angle estimation
    def get_localization(self):
        """
        returns the position of the drone
        """

        measured_position = self.measured_gps_position()
        self.drone_angle = self.measured_compass_angle()

        angle = self.measured_compass_angle() 

        rot_matrix = np.array([[math.cos(angle), math.sin(angle)],[-math.sin(angle), math.cos(angle)]])
        command = np.array([self.controller.command["forward"], self.controller.command["lateral"]])
        command = command@rot_matrix

        theorical_velocity = self.theorical_velocity + (command*0.6 - self.theorical_velocity*0.1)
        v = self.odometer_values()[0]

        if measured_position is not None and abs(v) > 5:  
            angle = self.measured_compass_angle() - self.angle_offset
            self.theorical_velocity = (np.array([v*math.cos(angle), v*math.sin(angle)]) + theorical_velocity) / 2
            theoretical_position = self.drone_position + self.theorical_velocity 
            self.drone_position = (self.measured_gps_position() + theoretical_position)/2
        elif measured_position is not None:
            angle = self.measured_compass_angle() - self.angle_offset
            self.theorical_velocity = np.array([v*math.cos(angle), v*math.sin(angle)]) / 2
            self.drone_position = self.measured_gps_position()
        else:
            self.theorical_velocity = theorical_velocity
            self.drone_position = self.drone_position + self.theorical_velocity


    def add_wounded(self, data_wounded):
        """
        compares the wounded persons detected with the ones already detected
        """

        def get_wounded_position():
            wounded_pos = self.drone_position.copy()
            angle = normalize_angle(self.drone_angle + data_wounded.angle)
            wounded_pos[0] += data_wounded.distance * math.cos(angle)
            wounded_pos[1] += data_wounded.distance * math.sin(angle)
            return wounded_pos

        wounded_pos = get_wounded_position()
    
        for k in range(len(self.wounded_found)):
            wounded = self.wounded_found[k]
            if np.linalg.norm(wounded_pos - wounded["position"]) < self.wounded_distance:
                wounded["count"] += 1
                n = wounded["count"]
                wounded["position"] = wounded["position"]*((n-1)/n) + wounded_pos/n
                wounded["last_seen"] = 0
                return
            
        self.wounded_found.append({"position": wounded_pos, "count": 1, "last_seen": 0})
    
    def clear_wounded_found(self):
        frame_limit = 10

        for k in range(len(self.wounded_found)-1,-1,-1):
            self.wounded_found[k]["last_seen"] += 1
            if np.linalg.norm(self.drone_position - self.wounded_found[k]["position"])<80 and self.wounded_found[k]["last_seen"] > frame_limit:
                self.wounded_found.pop(k)

    def process_semantic_sensor(self):
        """
        According to his state in the state machine, the Drone will move towards a wound person or the rescue center
        """

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        command = {"forward": 1.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angular_vel_controller_max = 1.0

        detection_semantic = self.semantic_values()
        
        best_angle = 0

        self.clear_wounded_found()

        found_wounded = False
        if (detection_semantic):
            for data in detection_semantic:
                # If the wounded person detected is held by nobody
                if data.entity_type == DroneSemanticSensor.TypeEntity.WOUNDED_PERSON and not data.grasped:
                    self.add_wounded(data)
            
            found_wounded = len(self.wounded_found) > 0
            if (len(self.wounded_found) > 0 and 
                (self.controller.current_state == self.controller.going_to_wounded 
                or self.controller.current_state == self.controller.approaching_wounded
                or self.controller.current_state == self.controller.roaming)):
            
                # Select the best one among wounded persons detected
                min_distance = np.linalg.norm(self.drone_position - self.wounded_found[0]["position"])
                best_position = self.wounded_found[0]["position"]
                for wounded in self.wounded_found[1:]:
                    distance = np.linalg.norm(self.drone_position - wounded["position"])
                    if distance < min_distance:
                        min_distance = distance
                        best_position = wounded["position"]
                best_angle = normalize_angle(angle(np.array([1,0]), best_position - self.drone_position))
                best_angle = normalize_angle(best_angle - normalize_angle(self.drone_angle))

        found_rescue_center = False
        is_near = False
        angles_list = []
        if (detection_semantic and 
            (self.controller.current_state == self.controller.going_to_center 
            or self.controller.current_state == self.controller.approaching_center)):
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
            return [[-320,-180],[-260,138],[-200,150], [20, -200]]
        
        path = self.map.shortest_path(self.drone_position, self.rescue_center_position)
        return path


    def get_control_from_path(self, pos):
        """
        returns the control to follow the path
        """

        command = {"forward": 0.0,
                   "lateral": 0.0,
                   "rotation": 0.0,
                   "grasper": 0}
        
        angle_from_waypoint = self.adapt_angle_direction(pos)
        angle_from_waypoint = normalize_angle(angle_from_waypoint+math.pi/4)

        if angle_from_waypoint > 0.5:
            command["rotation"] =  1.0
        elif angle_from_waypoint < -0.5:
            command["rotation"] =  -1.0
        else:
            command["rotation"] = angle_from_waypoint

        command["forward"] = math.cos(angle_from_waypoint-math.pi/4)
        command["lateral"] = math.sin(angle_from_waypoint-math.pi/4)
        norm = max(abs(command["forward"]),abs(command["lateral"]))
        command["forward"] = command["forward"]/norm
        command["lateral"] = command["lateral"]/norm     
        
        if self.nextWaypoint != None:
            if self.check_waypoint(pos):
                if len(self.path) > 0:
                    self.lastWaypoint = self.nextWaypoint.copy()
                    self.nextWaypoint = self.path.pop()
                else:
                    self.nextWaypoint = None
                    self.onRoute = False
        return command
    
    def compute_rescue_center_position(self):
        """
        computes the position of the rescue center
        """
        semantic_lidar_dist = [data.distance for data in self.semantic_values() if data.entity_type == DroneSemanticSensor.TypeEntity.RESCUE_CENTER]
        min_dist = min(semantic_lidar_dist) if len(semantic_lidar_dist) > 0 else np.inf
        
        if min_dist > 10:
            self.rescue_center_position = self.drone_position.copy()

    def control(self):

        self.found_wounded, self.found_center, self.command_semantic = self.process_semantic_sensor()
        self.get_localization()
        
        if self.rescue_center_position is None:
            self.compute_rescue_center_position()
        
        self.controller.cycle()

        self.update_mapping()
        self.keep_distance_from_walls()
            
        return self.controller.command
    
    def keep_distance_from_walls(self):
        """
        keeps the drone at a distance from the walls
        """
        
        lidar_dist = self.lidar().get_sensor_values()
        lidar_angles = self.lidar().ray_angles

        min_dist_index = np.argmin(lidar_dist)
        min_dist = lidar_dist[min_dist_index]
        min_angle = lidar_angles[min_dist_index]

        if min_dist < 30:
            return min_angle, min_dist
        return None
    
    def update_mapping(self):
        """
        updates the map
        """
        detection_semantic = self.semantic_values()
        self.estimated_pose = Pose(self.drone_position, self.drone_angle)
        max_vel_angle = 0.08
        if abs(self.measured_angular_velocity()) < max_vel_angle:
            self.map.update(self.estimated_pose, detection_semantic)


    def draw_top_layer(self):

        if self.debug_wounded:
            for wounded in self.wounded_found:
                pos = np.array(wounded["position"]) + np.array(self.size_area)/2
                arcade.draw_circle_filled(pos[0], pos[1],10, arcade.color.GREEN_YELLOW)
                arcade.draw_circle_outline(pos[0], pos[1],self.wounded_distance, arcade.color.GREEN_YELLOW)

        if self.debug_positions:
            pos = np.array(self.drone_position) + np.array(self.size_area)/2
            arcade.draw_circle_filled(pos[0], pos[1],5, arcade.color.RED)

            direction = np.array([1,0])
            rot = np.array([[math.cos(self.drone_angle), math.sin(self.drone_angle)],[-math.sin(self.drone_angle), math.cos(self.drone_angle)]])
            direction = direction@rot
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*200, pos[1]+direction[1]*200, arcade.color.RED)

            direction = self.theorical_velocity
            arcade.draw_line(pos[0], pos[1], pos[0]+direction[0]*20, pos[1]+direction[1]*20, arcade.color.GREEN)


    def draw_bottom_layer(self):

        if self.debug_path: 
            drawn_path = self.path.copy()
            if self.nextWaypoint != None: drawn_path.append(self.nextWaypoint)
            if self.lastWaypoint != None: drawn_path.append(self.lastWaypoint)
            for k in range(len(drawn_path)-1):
                pt1 = np.array(drawn_path[k]) + np.array(self.size_area)/2
                pt2 = np.array(drawn_path[k+1]) + np.array(self.size_area)/2
                arcade.draw_line(pt2[0], pt2[1], pt1[0], pt1[1], color=(255, 0, 255))