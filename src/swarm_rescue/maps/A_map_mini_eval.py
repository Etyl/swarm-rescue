"""
This file is an empty map file used by A_fast_image_to_map.py
to generate a map file.
Some parts of the file like map_mini_eval are replaced when a map is generated
"""

import math
import random
from typing import List, Type

from spg.playground import Playground
from spg.utils.definitions import CollisionTypes

from spg_overlay.entities.drone_abstract import DroneAbstract, drone_collision_wall, drone_collision_drone
from spg_overlay.entities.rescue_center import RescueCenter, wounded_rescue_center_collision
from spg_overlay.entities.sensor_disablers import ZoneType, NoGpsZone, srdisabler_disables_device, KillZone, NoComZone
from spg_overlay.entities.wounded_person import WoundedPerson
from spg_overlay.gui_map.closed_playground import ClosedPlayground
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig
from spg_overlay.utils.misc_data import MiscData

from .A_walls_map_mini_eval import add_walls, add_boxes


class MapMiniEval(MapAbstract):

    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._time_step_limit = 2000
        self._real_time_limit = 120

        # PARAMETERS MAP
        self._size_area = (1540, 1115)

        self._rescue_center = RescueCenter(size=(210,155))
        self._rescue_center_pos = ((637,-451), 0)

        self._no_gps_zone = NoGpsZone(size=(400, 600))
        self._no_gps_zone_pos = ((-130, 0), 0)

        self._kill_zone = KillZone(size=(100, 200))
        self._kill_zone_pos = ((-540, 60), 0)

        self._no_com_zone = NoComZone(size=(500, 300))
        self._no_com_zone_pos = ((600, 0), 0)

        self._wounded_persons_pos = [(67,-506), (-716,-504), (65,-254), (-488,-251), (-131,61), (303,63), (676,69), (-405,130), (267,225), (-720,505), (710,513), (54,512)]
        self._number_wounded_persons = len(self._wounded_persons_pos)
        self._wounded_persons: List[WoundedPerson] = []

        orient = random.uniform(-math.pi, math.pi)
        self._drones_pos = [((640, -350), orient), ((600, -350), orient), ((560, -350), orient), ((680, -350), orient), ((640, -300), orient), ((600, -300), orient), ((560, -300), orient), ((680, -300), orient)]
        self._number_drones = len(self._drones_pos)
        self._drones: List[DroneAbstract] = []

    def construct_playground(self, drone_type: Type[DroneAbstract]) -> Playground:
        playground = ClosedPlayground(size=self._size_area)

        # RESCUE CENTER
        playground.add_interaction(CollisionTypes.GEM,
                                   CollisionTypes.ACTIVABLE_BY_GEM,
                                   wounded_rescue_center_collision)

        playground.add(self._rescue_center, self._rescue_center_pos)

        add_walls(playground)
        add_boxes(playground)

        self._explored_map.initialize_walls(playground)

        # DISABLER ZONES
        playground.add_interaction(CollisionTypes.DISABLER,
                                   CollisionTypes.DEVICE,
                                   srdisabler_disables_device)

        if ZoneType.NO_GPS_ZONE in self._zones_config:
            playground.add(self._no_gps_zone, self._no_gps_zone_pos)

        if ZoneType.KILL_ZONE in self._zones_config:
            playground.add(self._kill_zone, self._kill_zone_pos)

        if ZoneType.NO_COM_ZONE in self._zones_config:
            playground.add(self._no_com_zone, self._no_com_zone_pos)

        # POSITIONS OF THE WOUNDED PERSONS
        for i in range(self._number_wounded_persons):
            wounded_person = WoundedPerson(rescue_center=self._rescue_center)
            self._wounded_persons.append(wounded_person)
            pos = (self._wounded_persons_pos[i], 0)
            playground.add(wounded_person, pos)

        # POSITIONS OF THE DRONES
        misc_data = MiscData(size_area=self._size_area,
                             number_drones=self._number_drones)
        for i in range(self._number_drones):
            drone = drone_type(identifier=i, misc_data=misc_data)
            self._drones.append(drone)
            playground.add(drone, self._drones_pos[i])

        playground.add_interaction(CollisionTypes.PART,
                                   CollisionTypes.ELEMENT,
                                   drone_collision_wall)
        playground.add_interaction(CollisionTypes.PART,
                                   CollisionTypes.PART,
                                   drone_collision_drone)

        return playground
