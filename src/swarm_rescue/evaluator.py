import gc
from typing import Tuple
import multiprocessing
import numpy as np
import time

from spg_overlay.entities.sensor_disablers import ZoneType
from spg_overlay.utils.constants import DRONE_INITIAL_HEALTH
from spg_overlay.reporting.evaluation import EvalConfig, EvalPlan, ZonesConfig
from spg_overlay.reporting.score_manager import ScoreManager
from spg_overlay.reporting.team_info import TeamInfo
from spg_overlay.gui_map.gui_sr import GuiSR

from maps.map_intermediate_01 import MyMapIntermediate01
from maps.map_intermediate_02 import MyMapIntermediate02
from maps.map_final_2023 import MyMapFinal
from maps.map_medium_01 import MyMapMedium01
from maps.map_medium_02 import MyMapMedium02

from solutions.my_drone_eval import MyDroneEval


"""
eval_config = EvalConfig(map_type=MyMapIntermediate01, nb_rounds=1)
self.eval_plan.add(eval_config=eval_config)

eval_config = EvalConfig(map_type=MyMapIntermediate02)
self.eval_plan.add(eval_config=eval_config)

zones_config: ZonesConfig = ()
eval_config = EvalConfig(map_type=MyMapMedium01, zones_config=zones_config, nb_rounds=1, config_weight=1)
self.eval_plan.add(eval_config=eval_config)

zones_config: ZonesConfig = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE, ZoneType.KILL_ZONE)
eval_config = EvalConfig(map_type=MyMapMedium01, zones_config=zones_config, nb_rounds=1, config_weight=1)
self.eval_plan.add(eval_config=eval_config)

zones_config: ZonesConfig = (ZoneType.NO_COM_ZONE, ZoneType.NO_GPS_ZONE, ZoneType.KILL_ZONE)
eval_config = EvalConfig(map_type=MyMapMedium02, zones_config=zones_config, nb_rounds=1, config_weight=1)
self.eval_plan.add(eval_config=eval_config)
"""


class MyDrone(MyDroneEval):
    pass


def multiprocessing_func(func):
    def wrapper(*args, **kwargs):
        try :
            func(*args, **kwargs)
        except :
            print("Error: function failed")
    return wrapper     

class Evaluator:
    """
    The Launcher class is responsible for running a simulation of drone rescue sessions. It creates an instance of the
    map with a specified zone type, constructs a playground using the construct_playground method of the map
    class, and initializes a GUI with the playground and map. It then runs the GUI, allowing the user to interact
    with it. After the GUI finishes, it calculates the score for the exploration of the map and saves the images and
    data related to the round.

    Fields
        nb_rounds: The number of rounds to run in the simulation.
        team_info: An instance of the TeamInfo class that stores team information.
        number_drones: The number of drones in the simulation.
        time_step_limit: The maximum number of time steps in the simulation.
        real_time_limit: The maximum elapsed real time in the simulation.
        number_wounded_persons: The number of wounded persons in the simulation.
        size_area: The size of the simulation area.
        score_manager: An instance of the ScoreManager class that calculates the final score.
        data_saver: An instance of the DataSaver class that saves images and data related to the simulation.
        video_capture_enabled: A boolean indicating whether video capture is enabled or not.
    """

    def __init__(self):

        """
        Here you can fill in the evaluation plan ("evalplan") yourself, adding or removing configurations.
        A configuration is defined by a map of the environment and whether or not there are zones of difficulty.
        """
        self.team_info = TeamInfo()

    def one_round(self, eval_config: EvalConfig):
        """
        The one_round method is responsible for running a single round of the session. It creates an instance of the
        map class with the specified eval_config, constructs a playground using the construct_playground method
        of the map class, and initializes a GUI with the playground and map. It then runs the GUI, which allows the
        user to interact with. After the GUI finishes, it calculates the score for the exploration of the map and saves
        the images and data related to the round.
        """
        my_map = eval_config.map_type(eval_config.zones_config)
        number_drones = my_map.number_drones
        time_step_limit = my_map.time_step_limit
        real_time_limit = my_map.real_time_limit
        number_wounded_persons = my_map.number_wounded_persons
        size_area = my_map.size_area

        self.score_manager = ScoreManager(number_drones=number_drones,
                                          time_step_limit=time_step_limit,
                                          real_time_limit=real_time_limit,
                                          total_number_wounded_persons=number_wounded_persons)

        playground = my_map.construct_playground(drone_type=MyDrone)

        my_gui = GuiSR(playground=playground,
                       the_map=my_map,
                       draw_interactive=False)

        my_map.explored_map.reset()

        # this function below is a blocking function until the round is finished
        my_gui.run()

        score_exploration = my_map.explored_map.score() * 100.0

        return (my_gui.percent_drones_destroyed,
                my_gui.mean_drones_health,
                my_gui.elapsed_time,
                my_gui.rescued_all_time_step,
                score_exploration, my_gui.rescued_number,
                my_gui.real_time_elapsed,
                my_gui.real_time_limit_reached)

    @multiprocessing_func
    def evaluate_single_drone(self,results, id):
        gc.disable()

        """
        The go method in the Launcher class is responsible for running the simulation for different eval_config,
         and calculating the score for each one.
        """

        eval_config = EvalConfig(map_type=MyMapIntermediate01, nb_rounds=1)
        my_map = eval_config.map_type(eval_config.zones_config)
        number_wounded_persons = my_map.number_wounded_persons

        gc.collect()
        print("")

        if not isinstance(eval_config.zones_config, Tuple) and not isinstance(eval_config.zones_config[0], Tuple):
            raise ValueError("Invalid eval_config.zones_config. It should be a tuple of tuples of ZoneType.")

        print(f"*** Map: {eval_config.map_name}, special zones: {eval_config.zones_name_casual}")

        gc.collect()
        result = self.one_round(eval_config)
        (percent_drones_destroyed, mean_drones_health, elapsed_time_step, rescued_all_time_step,
            score_exploration, rescued_number, real_time_elapsed, real_time_limit_reached) = result

        result_score = self.score_manager.compute_score(rescued_number,
                                                        score_exploration,
                                                        rescued_all_time_step)
        (round_score, percent_rescued, score_time_step) = result_score

        if real_time_limit_reached:
            return None
        
        results[id] = (round_score, score_exploration, rescued_number/number_wounded_persons, rescued_all_time_step)
  

def evaluate(number_processes: int = 16):
    manager = multiprocessing.Manager()
    return_dict = manager.dict()

    evaluator = Evaluator()

    jobs = []
    for i in range(number_processes//4):
        for j in range(4):
            proc = multiprocessing.Process(target=evaluator.evaluate_single_drone, args=(return_dict,4*i+j,))
            jobs.append(proc)
            proc.start()
    
        for proc in jobs:
            proc.join()

    results_avg = np.mean(return_dict.values(), axis=0)
    results_avg[0] /= 100
    results_avg[1] /= 100
    results_avg = list(results_avg)
    results_avg.append(len(return_dict.values())/(number_processes-number_processes%4))

    return results_avg


def main() -> None:

    t0 = time.perf_counter()
    results_avg = evaluate(8)
    t = time.perf_counter() - t0

    print(f"Score: {results_avg[0]:.3f}")
    print(f"Exploration: {results_avg[1]:.3f}")
    print(f"Rescued: {results_avg[2]:.3f}")
    print(f"Rescue time: {int(results_avg[3])}")
    print(f"Process success rate: {results_avg[4]:.3f}")
    print(f"Total time: {t:.1f}")

            
if __name__ == "__main__":
    main()
