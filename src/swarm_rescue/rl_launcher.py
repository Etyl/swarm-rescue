import os
import sys
import time
import uuid
import subprocess
from datetime import datetime
from multiprocessing import Pool

from maps.map_final_2022_23 import MyMapFinal2022_23
from maps.map_final_2023_24_01 import MyMapFinal_2023_24_01
from maps.map_final_2023_24_02 import MyMapFinal_2023_24_02
from maps.map_final_2023_24_03 import MyMapFinal_2023_24_03
from maps.map_medium_01 import MyMapMedium01
from maps.map_medium_02 import MyMapMedium02


def convert_seconds(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))


def run_simulation(task):
    """Run a single simulation task as a subprocess."""
    map_name, epsilon, file_path = task
    os.makedirs(file_path, exist_ok=True)
    python_path = os.path.dirname(os.path.abspath(__file__))
    python_path = os.path.join(python_path, "parallel_rl.py")
    subprocess.run(
        [
            sys.executable,
            python_path,  # Replace with your actual script name
            "--map",
            map_name,
            "--output",
            file_path,
        ]
    )


if __name__ == "__main__":
    maps = [
        MyMapFinal2022_23,
        MyMapFinal_2023_24_01,
        MyMapFinal_2023_24_02,
        MyMapFinal_2023_24_03,
        MyMapMedium01,
        MyMapMedium02,
    ]

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), f"rl-run_{date}")
    os.makedirs(main_dir, exist_ok=True)

    round = 20
    tasks = []
    for map_class in maps:
        for k in range(round):
            file_path = os.path.join(
                main_dir,
                f"{map_class.__name__}_{uuid.uuid4()}"
            )
            tasks.append((map_class.__name__, file_path))


    num_workers = min(len(tasks), 20)
    time_start = time.time()

    print(f"Starting {len(tasks)} tasks with {num_workers} workers.")

    with Pool(processes=num_workers) as pool:
        pool.map(run_simulation, tasks)

    print(f"Finished all tasks in {convert_seconds(time.time() - time_start)}")
    print(f"All results saved in: {main_dir}")
