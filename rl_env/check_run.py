import numpy as np
from solutions.utils.constants import FRONTIER_COUNT,FRONTIER_FEATURES

run = []
with open("run.txt", "r") as f:
    for line in f:
        run.append(list(map(float,line.strip().split())))

run = np.array(run)
print("ok")

