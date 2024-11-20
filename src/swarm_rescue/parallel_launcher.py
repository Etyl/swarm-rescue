import argparse
import gc
import os
import sys
import json
import traceback

from spg_overlay.entities.sensor_disablers import ZoneType
from spg_overlay.reporting.evaluation import EvalConfig, EvalPlan, ZonesConfig
from spg_overlay.reporting.score_manager import ScoreManager
from spg_overlay.reporting.data_saver import DataSaver
from spg_overlay.reporting.team_info import TeamInfo
from spg_overlay.gui_map.gui_sr import GuiSR
from spg_overlay.gui_map.map_abstract import MapAbstract

from maps_import import *

from solutions.my_drone_eval import MyDroneEval

class MyDrone(MyDroneEval):
    pass


class Launcher:
    def __init__(self, map_type: MapAbstract, team_info: TeamInfo, run_name: str, result_path: str, video_capture_enabled: bool = False, zone_type: ZoneType = None, nb_rounds: int = 1):

        self.team_info = team_info
        self.eval_plan = EvalPlan()
        self.run_name = run_name

        if zone_type is None:
            eval_config = EvalConfig(map_type=map_type, nb_rounds=nb_rounds)
        else:
            zones_config: ZonesConfig = (zone_type,)
            eval_config = EvalConfig(map_type=map_type, zones_config=zones_config, nb_rounds=nb_rounds)

        self.eval_plan.add(eval_config=eval_config)

        self.number_drones = None
        self.max_timestep_limit = None
        self.max_walltime_limit = None
        self.number_wounded_persons = None
        self.size_area = None

        self.score_manager = None
        stat_saving_enabled = False
        self.video_capture_enabled = video_capture_enabled

        self.result_path = result_path
        self.data_saver = DataSaver(team_info=self.team_info,
                                    result_path=self.result_path,
                                    enabled=stat_saving_enabled)

    def one_round(self, eval_config: EvalConfig, num_round: int, hide_solution_output: bool = False):

        my_map = eval_config.map_type(eval_config.zones_config)
        self.number_drones = my_map.number_drones
        self.max_timestep_limit = my_map.max_timestep_limit
        self.max_walltime_limit = my_map.max_walltime_limit
        self.number_wounded_persons = my_map.number_wounded_persons
        self.size_area = my_map.size_area

        self.score_manager = ScoreManager(number_drones=self.number_drones,
                                          max_timestep_limit=self.max_timestep_limit,
                                          max_walltime_limit=self.max_walltime_limit,
                                          total_number_wounded_persons=self.number_wounded_persons)

        my_playground = my_map.construct_playground(drone_type=MyDrone)

        num_round_str = str(num_round)
        if self.video_capture_enabled:
            try:
                os.makedirs(self.result_path + "/videos/", exist_ok=True)
            except FileExistsError as error:
                print(error)
            filename_video_capture = (f"{self.result_path}videos/"
                                      f"team{self.team_info.team_number_str}_"
                                      f"{eval_config.map_name}_"
                                      f"{eval_config.zones_name_for_filename}_"
                                      f"rd{num_round_str}"
                                      f".avi")
        else:
            filename_video_capture = None

        my_gui = GuiSR(playground=my_playground,
                       the_map=my_map,
                       draw_interactive=False,
                       filename_video_capture=filename_video_capture)

        window_title = (f"Team: {self.team_info.team_number_str}   -   "
                        f"Map: {type(my_map).__name__}   -   "
                        f"Round: {num_round_str}")
        my_gui.set_caption(window_title)

        my_map.explored_map.reset()

        has_crashed = False
        error_msg = ""

        if hide_solution_output:
            original_stdout = sys.stdout
            sys.stdout = open(os.devnull, 'w')

        try:
            my_gui.run()
        except Exception as error:
            error_msg = traceback.format_exc()
            my_gui.close()
            has_crashed = True
        finally:
            if hide_solution_output:
                sys.stdout.close()
                sys.stdout = original_stdout

        if has_crashed:
            print(error_msg)

        score_exploration = my_map.explored_map.score() * 100.0
        score_health_returned = my_map.compute_score_health_returned() * 100

        last_image_explo_lines = my_map.explored_map.get_pretty_map_explo_lines()
        last_image_explo_zones = my_map.explored_map.get_pretty_map_explo_zones()
        self.data_saver.save_images(my_gui.last_image,
                                    last_image_explo_lines,
                                    last_image_explo_zones,
                                    eval_config.map_name,
                                    eval_config.zones_name_for_filename,
                                    num_round)
        
        result_score = self.score_manager.compute_score(my_gui.rescued_number,
                                                        score_exploration,
                                                        score_health_returned,
                                                        my_gui.full_rescue_timestep)
        (round_score, percent_rescued, score_timestep) = result_score        

        return {
            "round": num_round,
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
            "has_crashed": has_crashed,
            "stdout_file": self.result_path + f"log.txt",
            "filename_video_capture": filename_video_capture,
        }

    def go(self, stop_at_first_crash: bool = False, hide_solution_output: bool = False):
        ok = True
        all_results = []

        for eval_config in self.eval_plan.list_eval_config:
            gc.collect()
            print(f"////////////////////////////////////////////////////////////////////////////////////////////")
            print(f"*** Map: {eval_config.map_name}, special zones: {eval_config.zones_name_casual}")
            
            map_results = {"map_name": eval_config.map_name, "rounds": []}

            for num_round in range(eval_config.nb_rounds):
                gc.collect()
                result = self.one_round(eval_config, num_round + 1, hide_solution_output)
                map_results["rounds"].append(result)

                if result["has_crashed"]:
                    print(f"\t* WARNING, this program has crashed!")
                    ok = False
                    if stop_at_first_crash:
                        self.data_saver.generate_pdf_report()
                        all_results.append(map_results)
                        return ok

            all_results.append(map_results)

        result_filename = f"{self.run_name}/results.json"

        try:
            with open(result_filename, "r") as result_file:
                results = json.load(result_file)
        except FileNotFoundError:
            results = []

        results += all_results

        with open(result_filename, "w") as result_file:
            json.dump(results, result_file, indent=4)

        self.data_saver.generate_pdf_report()
        return ok


if __name__ == "__main__":
    gc.disable()
    parser = argparse.ArgumentParser(description="Launcher of a swarm-rescue simulator for the competition")
    parser.add_argument("--map", "-m", required=True, help="Map to run")
    parser.add_argument("--name", "-n", required=True, help="Name of the run")
    parser.add_argument("--result_path", "-rp", required=True, help="Path to save the results")
    parser.add_argument("--video", "-v", required=True, help="Enable video capture")
    parser.add_argument("--zone", "-z", required=True, help="Zone type to run")
    parser.add_argument("--rounds", "-r", required=True, help="Number of rounds to run")
    args = parser.parse_args()

    map_class = globals()[args.map]
    team_info = TeamInfo()
    video_capture_enabled = bool(args.video)
    zone_type = getattr(ZoneType, args.zone, None)
    nb_rounds = int(args.rounds)
    result_path = args.result_path

    launcher = Launcher(map_type=map_class, team_info=team_info, run_name=args.name, result_path=result_path, video_capture_enabled=video_capture_enabled, zone_type=zone_type, nb_rounds=nb_rounds)
    ok = launcher.go()
    if not ok:
        exit(1)
