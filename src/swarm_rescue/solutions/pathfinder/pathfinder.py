import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pyastar2d

robot_radius = 10
sub_segment_size = 20
path_refinements = 3 # number of times to refine the path
save_images = True
output = "./solve"


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

def interpolate_path(point1, point2, t):
    return point1 + (point2-point1)*t

def is_path_free(map, point1, point2):
    n = int(np.sum(np.abs(point2-point1)))
    for t in range(n+2):
        if map[round(interpolate_path(point1, point2, t/(n+1))[0])][round(interpolate_path(point1, point2, t/(n+1))[1])] == True:
            return False
    return True

# TODO: C implementation 
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
    path_refined = smooth_path(map_border, path_refined)
    for _ in range(path_refinements-1):
        path_refined = segmentize_path(map_border, path_refined)
        path_refined = smooth_path(map_border, path_refined)
    print(f"Refined path of length {len(path_refined)} in {time.time()-t2:.6f}s")

    if len(path_refined) > 0:        
        if save_images:
            plt.imshow(np.stack((map_border, map_border, map_border), axis=2).astype(np.float32))
            plt.savefig("./border.png")
            map = np.stack((map, map, map), axis=2)

            current_output = output+"-raw.png"
            plt.figure()
            path_plot = np.array(path)
            plt.plot(path_plot[:,1],path_plot[:,0],color="red")
            plt.imshow(map)
            print(f"Plotting path to {current_output}, {len(path)} points)")
            plt.savefig(current_output)

            current_output = output+"-smooth.png"
            plt.figure()
            path_plot = np.array(path_smooth)
            plt.plot(path_plot[:,1],path_plot[:,0],color="red")
            plt.imshow(map)
            print(f"Plotting path to {current_output}, {len(path_smooth)} points)")
            plt.savefig(current_output)

            current_output = output+"-refined.png"
            plt.figure()
            path_plot = np.array(path_refined)
            plt.plot(path_plot[:,1],path_plot[:,0],color="red")
            plt.imshow(map)
            print(f"Plotting path to {current_output}, {len(path_refined)} points)")
            plt.savefig(current_output)

    return path_refined


map = mpimg.imread("./map-f.png").astype(np.float32)
map = (map[:, :, 0]+map[:, :, 1]+map[:, :, 3])/3
start = np.array([30, 50])
end = np.array([600, 800])

# input map with 1=free, 0=obstacle
pathfinder(map, start, end)
