import matplotlib.pyplot as plt # type: ignore
import pickle
from typing import Optional
import numpy as np
from solutions.pathfinder.pathfinder import pathfinder # type: ignore
import os

def get_maze(loc: str) -> Optional[np.ndarray]:
    with open(loc, 'rb') as f:
        maze = pickle.load(f)
        return maze


def plot_pathfinder(loc:str, start, end, radius):
    maze = get_maze(loc)
    if maze is None:
        return
    path = pathfinder(maze, start, end, radius)
    plt.figure()
    plt.imshow(maze, cmap='Greys')
    plt.plot(path[:,1], path[:,0], color='red')


def main() -> None:
    path: str = os.path.dirname(os.path.abspath(__file__))
    plot_pathfinder(path + "/data/small_maze.pkl",[17,27], [45, 176], 5)
    plot_pathfinder(path + "/data/medium_maze.pkl",[884, 108], [918, 964], 20)
    plt.show()

if __name__ == '__main__':
    main()