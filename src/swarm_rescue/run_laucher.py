import sys
import threading
import subprocess
import time
from spg_overlay.reporting.result_path_creator import ResultPathCreator
from spg_overlay.reporting.team_info import TeamInfo
import uuid
import os

def launch_parallel_task(
    script_name, map_name, name, result_path, zone_type, nb_rounds, video_capture_enabled=False
):
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
                "--rounds",
                str(nb_rounds),
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
    map_names = [
        ("MyMapIntermediate01", "NO_GPS_ZONE", 3),
        ("MyMapIntermediate01", "NONE", 2),
        ("MyMapIntermediate01", "NO_GPS_ZONE", 1),
    ]
    video_capture_enabled = True

    script_name = "src/swarm_rescue/parallel_launcher.py"

    threads = []
    for map_name, zone_type, nb_rounds in map_names:
        # random id to avoid conflicts
        result_path = (
            name 
            + "/"
            + str(uuid.uuid4())
            + "_map_"
            + map_name
            + "_zone_"
            + zone_type
            + "_rounds_"
            + str(nb_rounds)
            + "/"
        )
        thread = threading.Thread(
            target=launch_parallel_task,
            args=(
                script_name,
                map_name,
                name,
                result_path,
                zone_type,
                nb_rounds,
                video_capture_enabled,
            ),
        )
        threads.append(thread)
        thread.start()

    for thread in threads:
        thread.join()

    print("All run information saved in: ", name)
