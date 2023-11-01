import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pyastar2d

robot_radius = 5
save_images = True
output = "./solve.png"


def border_from_map_np(map):
    map = np.ones(map.shape).astype(np.float32)-map
    for _ in range(robot_radius):
        map = (np.roll(map,(1, 0), axis=(1, 0))+
               np.roll(map,(0, 1), axis=(1, 0))+
               np.roll(map,(-1, 0), axis=(1, 0))+
               np.roll(map,(0, -1), axis=(1, 0))+
               np.roll(map,(1, 1), axis=(1, 0))+
               np.roll(map,(-1, -1), axis=(1, 0))+
               np.roll(map,(1, -1), axis=(1, 0))+
               np.roll(map,(-1, 1), axis=(1, 0)))
    return map>0.5

# TODO : optimize with numpy roll
def border_from_map(map):
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
            if count>=round(robot_radius):
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
    border_map = np.array([[False for _ in range(len(map[0]))]
                            for _ in range(len(map))])
    for ix in range(len(map)):
        for iy in range(len(map[0])):
            if dist_map[ix][iy] > -0.5:
                border_map[ix][iy] = True

    return border_map

def interpolate_path(point1, point2, t):
    return point1 + (point2-point1)*t

def is_path_free(map, point1, point2):
    n = np.sum(np.abs(point2-point1))
    for t in range(n+2):
        if map[round(interpolate_path(point1, point2, t/(n+1))[0])][round(interpolate_path(point1, point2, t/(n+1))[1])] == True:
            return False
    return True

def smooth_path(map, path):
    i_ref = 0
    j_ref = 2
    new_path = [path[0]]
    while i_ref<len(path):
        if j_ref<len(path) and is_path_free(map, path[i_ref], path[j_ref]):
            j_ref += 1
        elif j_ref>=len(path):
            new_path.append(path[-1])
            break
        else:
            new_path.append(path[j_ref-1])
            i_ref = j_ref-1
            j_ref = i_ref+2
    if new_path[-1][0]!=path[-1][0] or new_path[-1][1]!=path[-1][1]:
        new_path.append(path[-1])
    return new_path



def pathfinder(map, start, end):

    tb = time.time()
    map_border = border_from_map_np(map)
    print(f"Border map generated in {time.time()-tb:.6f}s")

    grid = np.ones(map.shape).astype(np.float32)
    grid[map_border == True] = np.inf

    assert grid.min() == 1, "cost of moving must be at least 1"

    t0 = time.time()
    # set allow_diagonal=True to enable 8-connectivity
    path = pyastar2d.astar_path(grid, start, end, allow_diagonal=False)
    dur = time.time() - t0
    print(f"Found path of length {path.shape[0]} in {dur:.6f}s")

    t1 = time.time()
    path = smooth_path(map_border, path)
    dur_smooth = time.time() - t1
    print(f"Smoothed path of length {len(path)} in {dur_smooth:.6f}s")

    if len(path) > 0:
        print(f"Path found")
        
        if save_images:
            map = np.stack((map, map, map), axis=2)
            path_plot = np.array(path)
            plt.plot(path_plot[:,1],path_plot[:,0],color="red")
            plt.imshow(map)
            print(f"Plotting path to {output}")
            plt.savefig(output)
    else:
        print("No path found")
    
    return path


map = mpimg.imread("./map-f.png").astype(np.float32)
map = map[:, :, 1]
start = np.array([30, 50])
end = np.array([600, 800])

# input map with 1=free, 0=obstacle
pathfinder(map, start, end)
