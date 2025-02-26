import json
import os.path
import traceback
from typing import Optional, List, Tuple
import numpy as np
from functools import partial
import os
import argparse
import gc

from maps.map_final_2022_23 import MyMapFinal2022_23 # type: ignore
from solutions.frontier_drone import FrontierDrone # type: ignore
from solutions.utils.constants import OBSERVATION_SPACE, ACTION_SPACE # type: ignore
from spg_overlay.gui_map.gui_sr import GuiSR # type: ignore
from spg_overlay.gui_map.map_abstract import MapAbstract # type: ignore
from spg_overlay.reporting.evaluation import ZonesConfig, EvalConfig # type: ignore
from spg_overlay.reporting.score_manager import ScoreManager # type: ignore


def get_run(
    save_dir: str,
    map_type: MapAbstract,
    zones_config: ZonesConfig = ()) -> None:
    """
    Gets samples from a run and saves them to a file
    """
    eval_config = EvalConfig(map_type=map_type, zones_config=zones_config)

    my_map = eval_config.map_type(eval_config.zones_config)

    my_playground = my_map.construct_playground(drone_type=partial(FrontierDrone, save_dir=save_dir))

    my_gui = GuiSR(playground=my_playground,
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

    score_exploration = my_map.explored_map.score() * 100.0
    score_health_returned = my_map.compute_score_health_returned() * 100

    score_manager = ScoreManager(
        number_drones=my_map.number_drones,
        max_timestep_limit=my_map.max_timestep_limit,
        max_walltime_limit=my_map.max_walltime_limit,
        total_number_wounded_persons=my_map.number_wounded_persons
    )

    result_score = score_manager.compute_score(my_gui.rescued_number,
                                                    score_exploration,
                                                    score_health_returned,
                                                    my_gui.full_rescue_timestep)
    (round_score, percent_rescued, score_timestep) = result_score

    saved_info = {
        "score": round_score,
        "percent_rescued": percent_rescued,
        "score_timestep": score_timestep,
        "percent_drones_destroyed": my_gui.percent_drones_destroyed,
        "mean_drones_health": my_gui.mean_drones_health,
        "elapsed_timestep": my_gui.elapsed_timestep,
        "full_rescue_timestep": my_gui.full_rescue_timestep,
        "score_exploration": score_exploration,
        "rescued_number": my_gui.rescued_number,
        "score_health_returned": score_health_returned,
        "elapsed_walltime": my_gui.elapsed_walltime,
        "is_max_walltime_limit_reached": my_gui.is_max_walltime_limit_reached,
        "has_crashed": has_crashed
    }

    if has_crashed:
        save_file_crash = os.path.join(save_dir, "crash.log")
        with open(save_file_crash,"w") as f:
            f.write(error_msg)

