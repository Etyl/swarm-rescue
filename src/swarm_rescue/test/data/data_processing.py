from typing import Optional

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import numpy as np
import pickle

def process_image(loc: str, out:str) -> None:
    img = mpimg.imread(loc)
    arr = np.zeros(img.shape[:2])
    for i in range(arr.shape[0]):
        for j in range(arr.shape[1]):
            if img[i,j,0]>0.5:
                arr[i,j] = 2
    with open(out, 'wb') as f:
        pickle.dump(arr, f)

def load_array(loc: str) -> Optional[np.ndarray]:
    with open(loc, 'rb') as f:
        arr = pickle.load(f)
        return arr

def main() -> None:
    process_image("small_maze.png", "small_maze.pkl")
    process_image("medium_maze.png", "medium_maze.pkl")



if __name__ == '__main__':
    main()
