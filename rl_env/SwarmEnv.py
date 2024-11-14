import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import arcade
import cv2
from functools import partial


from maps.map_final_2023_24_01 import MyMapFinal_2023_24_01
from solutions.frontier_drone import FrontierDrone
from solutions.utils.constants import FRONTIER_COUNT, FRONTIER_FEATURES
from spg_overlay.entities.sensor_disablers import ZoneType
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.reporting.data_saver import DataSaver
from spg_overlay.reporting.evaluation import EvalPlan, ZonesConfig, EvalConfig
from spg_overlay.reporting.result_path_creator import ResultPathCreator
from spg_overlay.reporting.score_manager import ScoreManager
from spg_overlay.reporting.team_info import TeamInfo


def get_run(
    policy,
    save_file_run,
    map_type = MyMapFinal_2023_24_01,
    zones_config: ZonesConfig = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE, ZoneType.KILL_ZONE)):
    """
    Gets samples from a run
    Params:
        - policy: policy used for run, function(state)->action_distribution
    Return:
        - run: [(state0,actions0,state0',reward0),...]
    """
    eval_config = EvalConfig(map_type=map_type, zones_config=zones_config)

    my_map = eval_config.map_type(eval_config.zones_config)

    my_playground = my_map.construct_playground(drone_type=partial(FrontierDrone, policy=policy, save_run=save_file_run))

    my_gui = GuiSR(playground=my_playground,
                   the_map=my_map,
                   draw_interactive=False)

    window_title = f"Map: {type(my_map).__name__}"
    my_gui.set_caption(window_title)

    my_map.explored_map.reset()

    has_crashed = False
    error_msg = ""

    try:
        # this function below is a blocking function until the round is finished
        my_gui.run()
    except Exception as error:
        error_msg = traceback.format_exc()
        my_gui.close()
        has_crashed = True

    if has_crashed:
        print(error_msg)


def dummy_policy(input:np.ndarray):
    if len(input) != FRONTIER_COUNT*FRONTIER_FEATURES:
        raise "Wrong input shape"
    return np.ones(FRONTIER_COUNT)/FRONTIER_COUNT

if __name__ == "__main__":
    get_run(dummy_policy, "run.txt")
