from maps.map_final_2022_23 import MyMapFinal2022_23
from maps.map_final_2023_24_01 import MyMapFinal_2023_24_01
from maps.map_final_2023_24_02 import MyMapFinal_2023_24_02
from maps.map_final_2023_24_03 import MyMapFinal_2023_24_03
from spg_overlay.reporting.evaluation import ZonesConfig


class LargeMap01(MyMapFinal2022_23):
    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._wounded_persons_pos = []
        self._number_wounded_persons = 0

class LargeMap02(MyMapFinal_2023_24_01):
    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._wounded_persons_pos = []
        self._number_wounded_persons = 0

class LargeMap03(MyMapFinal_2023_24_02):
    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._wounded_persons_pos = []
        self._number_wounded_persons = 0

class LargeMap04(MyMapFinal_2023_24_03):
    def __init__(self, zones_config: ZonesConfig = ()):
        super().__init__(zones_config)
        self._wounded_persons_pos = []
        self._number_wounded_persons = 0




