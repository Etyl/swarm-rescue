import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pyastar2d

robot_radius = 10
sub_segment_size = 20
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
    n = int(np.sum(np.abs(point2-point1)))
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

def segmentize(path, i):
    num_sub_segments = round(np.sqrt(np.sum((path[i+1]-path[i])**2))/sub_segment_size)
    return [np.rint(interpolate_path(path[i], path[i+1], t/(num_sub_segments+1))) for t in range(num_sub_segments+2)]

def segmentize_path(map, path):
    new_path = []
    currentSegment = []
    nextSegment = segmentize(path, 0)
    start_point, end_point, max_segment = 0, 0, np.inf
    for k in range(len(path)-1):
        currentSegment = nextSegment[end_point:]
        if k<len(path)-2:
            nextSegment = segmentize(path, k+1)
        start_point, end_point, max_segment = 0, 0, 0
        for i in range(len(currentSegment)-1):
            for j in range(len(nextSegment)-1,-1,-1):
                if is_path_free(map, currentSegment[i], nextSegment[j]):
                    val = (np.sqrt(np.sum((currentSegment[i]-nextSegment[0])**2))+
                           np.sqrt(np.sum((nextSegment[0]-nextSegment[j])**2))-
                           np.sqrt(np.sum((currentSegment[i]-nextSegment[j])**2)))
                    if val>max_segment:
                        start_point = i
                        end_point = j
                        max_segment = val
                        break
        
        new_path.append(currentSegment[0])
        if max_segment>0 and start_point!=0:
            new_path.append(currentSegment[start_point])

    return new_path

# TODO: subdviser chaque segment du smooth path est essayer de coller au segment suivant pour CV vers meilleur solution
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
    path_smooth = smooth_path(map_border, path)
    dur_smooth = time.time() - t1
    print(f"Smoothed path of length {len(path_smooth)} in {dur_smooth:.6f}s")

    t2 = time.time()
    path_refined = segmentize_path(map_border, path_smooth)
    print(f"Refined path of length {len(path_refined)} in {time.time()-t2:.6f}s")

    if len(path_refined) > 0:
        print(f"Path found")
        
        if save_images:
            plt.imshow(np.stack((map_border, map_border, map_border), axis=2).astype(np.float32))
            plt.savefig("./border.png")

            plt.figure()
            map = np.stack((map, map, map), axis=2)
            path_plot = np.array(path_refined)
            plt.plot(path_plot[:,1],path_plot[:,0],color="red")
            plt.imshow(map)
            print(f"Plotting path to {output}, {len(path_refined)} points)")
            plt.savefig(output)
    else:
        print("No path found")
    
    return path


map = mpimg.imread("./map-f.png").astype(np.float32)
map = (map[:, :, 0]+map[:, :, 1]+map[:, :, 3])/3
start = np.array([30, 50])
end = np.array([600, 800])

# input map with 1=free, 0=obstacle
pathfinder(map, start, end)
