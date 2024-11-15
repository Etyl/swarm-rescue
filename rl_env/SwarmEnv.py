import traceback
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import arcade
import cv2
from functools import partial

from spg.playground import Playground
from spg.utils.definitions import PYMUNK_STEPS

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



class PlaygroundWrapperRL:
    def __init__(self, playground, my_map, exploration_scores):
        self._playground = playground
        self._map = my_map
        self.exploration_scores = exploration_scores

    def step(self, *args, **kwargs):
        self.exploration_scores.append(self._map.explored_map.score())
        return self._playground.step(*args, **kwargs)

    def __getattr__(self, name):
        # Forward any other attribute calls to the wrapped playground
        return getattr(self._playground, name)


def get_run(
    policy,
    save_file_run,
    map_type = MyMapFinal_2023_24_01,
    zones_config: ZonesConfig = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE, ZoneType.KILL_ZONE)):
    """
    Gets samples from a run and saves them to a file, each row being:
    state (FRONTER_COUNT*FRONTIER_FEATURES), action (FRONITER_COUNT), next_state (FRONTER_COUNT*FRONTIER_FEATURES), reward (1)
    Params:
        - policy: policy used for run, function(state)->action_distribution
    """
    eval_config = EvalConfig(map_type=map_type, zones_config=zones_config)

    my_map = eval_config.map_type(eval_config.zones_config)

    exp_score = []
    save_run = []
    my_playground = my_map.construct_playground(drone_type=partial(FrontierDrone, policy=policy, save_run=save_run))
    rl_playground = PlaygroundWrapperRL(my_playground, my_map, exp_score)

    my_gui = GuiSR(playground=rl_playground,
                   the_map=my_map,
                   draw_interactive=False)

    window_title = f"Map: {type(my_map).__name__}"
    my_gui.set_caption(window_title)

    my_map.explored_map.reset()

    has_crashed = False
    error_msg = ""

    try:
        my_gui.run()
    except Exception as error:
        error_msg = traceback.format_exc()
        my_gui.close()
        has_crashed = True

    reward_idx = -1
    with open(save_file_run,"w") as f:
        for s in save_run:
            s[reward_idx] = exp_score[min(int(s[reward_idx])-1, len(exp_score)-1)]
            f.write(" ".join(map(str,s)) + "\n")

    if has_crashed:
        print(error_msg)


def dummy_policy(input:np.ndarray):
    if len(input) != FRONTIER_COUNT*FRONTIER_FEATURES:
        raise "Wrong input shape"
    return np.ones(FRONTIER_COUNT)/FRONTIER_COUNT

if __name__ == "__main__":
    get_run(dummy_policy, "run0.txt")
