import argparse
import numpy as np
import matplotlib.image as mpimg
import time

import pyastar2d

from os.path import basename, join


def parse_args():
    parser = argparse.ArgumentParser(
        "An example of using pyastar2d to find the solution to a maze"
    )
    parser.add_argument(
        "--input", type=str, default="mazes/maze_small.png",
        help="Path to the black-and-white image to be used as input.",
    )
    parser.add_argument(
        "--output", type=str, help="Path to where the output will be written",
    )

    args = parser.parse_args()

    if args.output is None:
        args.output = join("solns", basename(args.input))

    return args

def calc_obstacle_map_from_map(map, rr):

    def norm(i,j,x,y):
        return np.sqrt((i-x)**2+(j-y)**2)

    def getNeighbours(x, y):
        neighbours = []
        for i in range(-1, 2):
            for j in range(-1, 2):
                if x+i >= 0 and y+j >= 0 and x+i < len(map) and y+j < len(map[0]) and (i != 0 or j != 0):
                    neighbours.append((x + i, y + j))
        return neighbours

    dist_map = map*(-1)
    new_dist_map = dist_map.copy()
    waiting_list = []
    for ix in range(len(map)):
        for iy in range(len(map[0])):
            if dist_map[ix][iy]>=0:
                for [nx, ny] in getNeighbours(ix, iy):
                    if dist_map[nx][ny]<-0.5:
                        if new_dist_map[nx][ny]<-0.5:
                            new_dist_map[nx][ny] = dist_map[ix][iy]+norm(ix,iy,nx,ny)
                            waiting_list.append((nx, ny))
                        else:
                            new_dist_map[nx][ny] = min(dist_map[ix][iy]+norm(ix,iy,nx,ny), new_dist_map[nx][ny])
                        
    dist_map = new_dist_map.copy()
    new_waiting_list = []
    count = 0
    while len(waiting_list)>0 or len(new_waiting_list)>0:        
        if len(waiting_list)==0:
            count += 1
            if count>=round(rr):
                break
            waiting_list = new_waiting_list.copy()
            new_waiting_list = []
        ix, iy = waiting_list.pop()
        for [nx, ny] in getNeighbours(ix, iy):
            if dist_map[nx][ny]<-0.5:
                dist_map[nx][ny] = dist_map[ix][iy]+norm(ix,iy,nx,ny)
                new_waiting_list.append((nx, ny))
            else:
                dist_map[nx][ny] = min(dist_map[ix][iy]+norm(ix,iy,nx,ny), dist_map[nx][ny])
            
    # obstacle map generation
    obstacle_map = np.array([[False for _ in range(len(map[0]))]
                            for _ in range(len(map))])
    for ix in range(len(map)):
        for iy in range(len(map[0])):
            if dist_map[ix][iy] > -0.5:
                obstacle_map[ix][iy] = True

    return obstacle_map


def main(map):   

    map_border = calc_obstacle_map_from_map(map, 6) 

    grid = np.ones(map.shape).astype(np.float32)
    grid[map_border == True] = np.inf

    assert grid.min() == 1, "cost of moving must be at least 1"

    # start is the first white block in the top row
    start = np.array([30, 50])

    # end is the first white block in the final column
    end = np.array([600, 800])

    t0 = time.time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar2d.astar_path(grid, start, end, allow_diagonal=False)
    dur = time.time() - t0

    if path.shape[0] > 0:
        print(f"Found path of length {path.shape[0]} in {dur:.6f}s")
        map = np.stack((map, map, map), axis=2)
        map[path[:, 0], path[:, 1]] = (1.0, 0, 0)

        print(f"Plotting path to {output}")
        mpimg.imsave(output, map.astype(np.float32))
    else:
        print("No path found")

    print("Done")

map = mpimg.imread("./map-f.png").astype(np.float32)
map = map[:, :, 1]
output = "./solve.png"


# input map with 1=free, 0=obstacle
main(map)
