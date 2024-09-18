import os
import pickle

import numpy as np
from typing import Optional, Any
from solutions.pathfinder.pathfinder import border_from_map_np # type: ignore

def get_maze(loc: str) -> Optional[np.ndarray]:
    with open(loc, 'rb') as f:
        maze = pickle.load(f)
        return maze

def test_small_border_map(benchmark: Any) -> None:
    path:str = os.path.dirname(os.path.abspath(__file__))
    maze = get_maze(path+"/data/small_maze.pkl")
    benchmark(lambda: border_from_map_np(maze, 10))
    assert benchmark.stats.stats.mean < 0.003


def test_medium_border_map(benchmark: Any) -> None:
    path: str = os.path.dirname(os.path.abspath(__file__))
    maze = get_maze(path + "/data/medium_maze.pkl")
    benchmark(lambda: border_from_map_np(maze, 10))
    assert benchmark.stats.stats.mean < 0.5

