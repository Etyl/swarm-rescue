import numpy as np

run = []
with open("run.txt", "r") as f:
    for line in f:
        run.append(list(map(float,line.strip().split())))

run = np.array(run)
print("ok")

