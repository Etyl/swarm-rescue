import traceback
from typing import Optional, List, Tuple
import numpy as np
from functools import partial

from rl_env.MapsRL import LargeMap02, LargeMap01
from solutions.frontier_drone import FrontierDrone
from solutions.utils.constants import OBSERVATION_SPACE, ACTION_SPACE
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract
from spg_overlay.reporting.evaluation import ZonesConfig, EvalConfig



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
    map_type: MapAbstract = LargeMap01,
    zones_config: ZonesConfig = ()) -> None:
    """
    Gets samples from a run and saves them to a file, each row being:
    state (OBSERVATION_SPACE), action (ACTION_SPACE), next_state (OBSERVATION_SPACE), reward (1)
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


def dummy_policy(obs:np.ndarray):
    if len(obs) != OBSERVATION_SPACE:
        raise "Wrong input shape"
    return np.ones(ACTION_SPACE)/ACTION_SPACE

if __name__ == "__main__":
    get_run(dummy_policy, "run0.txt", map_type=LargeMap02)
