import uuid
from datetime import datetime
import os
import time
import multiprocessing
import sys

from policies import deterministic_policy,epsilon_greedy_wrapper
from rl_env import get_run_wrapped

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../../src/swarm_rescue')))

from maps.map_final_2022_23 import MyMapFinal2022_23
from maps.map_final_2023_24_01 import MyMapFinal_2023_24_01
from maps.map_final_2023_24_02 import MyMapFinal_2023_24_02
from maps.map_final_2023_24_03 import MyMapFinal_2023_24_03
from maps.map_medium_01 import MyMapMedium01
from maps.map_medium_02 import MyMapMedium02



def convert_seconds(seconds):
    return time.strftime("%H:%M:%S", time.gmtime(seconds))

if __name__ == '__main__':

    maps = [
        MyMapFinal2022_23,
        MyMapFinal_2023_24_01,
        MyMapFinal_2023_24_02,
        MyMapFinal_2023_24_03,
        MyMapMedium01,
        MyMapMedium02,
    ]

    date = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    main_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "rl-run_"+date)

    if not os.path.exists(main_dir):
        os.makedirs(main_dir)

    number_rounds = 100
    tasks = []
    for map in maps:
        epsilon = lambda t: 0.5 * (number_rounds-1-t)/(number_rounds-1)
        for k in range(number_rounds):
            file_path  = os.path.join(main_dir, map.__name__+"_"+str(round(epsilon(k),3))+"_"+str(uuid.uuid4()))
            if not os.path.exists(file_path):
                os.makedirs(file_path)
            policy = epsilon_greedy_wrapper(epsilon(k))
            tasks.append((policy,file_path,map))


    num_workers = min(len(tasks), 10)

    time_start = time.time()

    print(f"Starting tasks, saved in {main_dir}")

    with multiprocessing.Pool(processes=num_workers) as pool:
        pool.map(get_run_wrapped, tasks)

    print(f"Finished tasks in {convert_seconds(time.time() - time_start)}")

