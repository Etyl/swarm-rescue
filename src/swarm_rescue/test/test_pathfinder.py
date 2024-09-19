import os
import pickle

import numpy as np
from typing import Optional, Any

import pytest

from solutions.pathfinder.pathfinder import pathfinder # type: ignore


def get_maze(loc: str) -> Optional[np.ndarray]:
    with open(loc, 'rb') as f:
        maze = pickle.load(f)
        return maze

def test_small_border_map(benchmark: Any) -> None:
    path:str = os.path.dirname(os.path.abspath(__file__))
    maze = get_maze(path+"/data/small_maze.pkl")
    benchmark(lambda: pathfinder(maze, [17,27], [45, 176], 5))


@pytest.mark.skip(reason="Too slow for now")
def test_medium_border_map(benchmark: Any) -> None:
    path: str = os.path.dirname(os.path.abspath(__file__))
    maze = get_maze(path + "/data/medium_maze.pkl")
    benchmark(lambda: pathfinder(maze, [884, 108], [918, 964]))