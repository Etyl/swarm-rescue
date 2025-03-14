import argparse
import gc
import os

from maps.map_final_2022_23 import MyMapFinal2022_23
from maps.map_final_2023_24_01 import MyMapFinal_2023_24_01
from maps.map_final_2023_24_02 import MyMapFinal_2023_24_02
from maps.map_final_2023_24_03 import MyMapFinal_2023_24_03
from maps.map_medium_01 import MyMapMedium01
from maps.map_medium_02 import MyMapMedium02
from rl_env.rl_env import get_run

if __name__ == "__main__":
    gc.disable()
    parser = argparse.ArgumentParser(description="RL launcher")
    parser.add_argument("--map", "-m", required=True, help="Map to run")
    parser.add_argument("--output", "-rp", required=True, help="Path to save the results")
    args = parser.parse_args()

    maps = [
        MyMapFinal2022_23,
        MyMapFinal_2023_24_01,
        MyMapFinal_2023_24_02,
        MyMapFinal_2023_24_03,
        MyMapMedium01,
        MyMapMedium02
    ]

    maps_dict = {map.__name__: map for map in maps}

    map_type = maps_dict[args.map]

    output_path = args.output

    get_run(output_path, map_type)
