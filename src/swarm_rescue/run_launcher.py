import sys
import threading
import subprocess
import time
from spg_overlay.reporting.result_path_creator import ResultPathCreator
from spg_overlay.reporting.team_info import TeamInfo
import uuid
import os
import multiprocessing

def launch_parallel_task(args):
    script_name, map_name, name, result_path, zone_type, video_capture_enabled = args
    os.makedirs(result_path, exist_ok=True)
    log_file_path = os.path.join(result_path, "log.txt")
    with open(log_file_path, "w") as log_file:
        subprocess.run(
            [
                sys.executable,
                script_name,
                "--map",
                map_name,
                "--name",
                name,
                "--zone",
                zone_type,
                "--video",
                str(video_capture_enabled),
                "--result_path",
                result_path,
            ],
            stdout=log_file,
    )


if __name__ == "__main__":
    team_info = TeamInfo()
    rpc = ResultPathCreator(team_info)

    name = rpc.path

    round_count = 1
    map_names = [
        ("MyMapFinal2022_23", "NONE"),
        ("MyMapFinal2022_23", "NO_GPS_ZONE"),
        ("MyMapFinal2022_23", "NO_COM_ZONE"),
        ("MyMapFinal2022_23", "KILL_ZONE"),
        ("MyMapFinal_2023_24_01", "NONE"),
        ("MyMapFinal_2023_24_02", "NONE"),
        ("MyMapFinal_2023_24_03", "NONE"),
        ("MyMapFinal_2023_24_01", "NO_COM_ZONE"),
        ("MyMapFinal_2023_24_02", "KILL_ZONE"),
        ("MyMapFinal_2023_24_03", "NO_GPS_ZONE")
    ]

    complete_map_names = []
    for map_name in map_names:
        complete_map_names = complete_map_names + [map_name]*round_count
    video_capture_enabled = True

    script_name = "src/swarm_rescue/parallel_launcher.py"

    # Prepare task arguments
    tasks = []
    for map_name, zone_type in complete_map_names:
        result_path = (
            name
            + "/"
            + str(uuid.uuid4())
            + "_map_"
            + map_name
            + "_zone_"
            + zone_type
            + "/"
        )
        tasks.append((
            script_name,
            map_name,
            name,
            result_path,
            zone_type,
            video_capture_enabled,
        ))

    num_workers = min(len(tasks), 10)

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(launch_parallel_task, tasks)

    print("All run information saved in: ", name)
