import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pyastar2d
import cv2

robot_radius = 10
sub_segment_size = 20
path_refinements = 3 # number of times to refine the path
save_images = False
output = "./solve"


def border_from_map_np(map, robot_radius):
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
    if save_images:
        plt.imshow(map>0.5)
        plt.savefig("./map_border.png")
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

    if new_path[-1][0]!=path[-1][0] or new_path[-1][1]!=path[-1][1]:
        new_path.append(path[-1])

    return new_path

def neighbors(map,point):
    neighbors = []
    if point[0]>0:
        neighbors.append((point[0]-1,point[1]))
    if point[0]<map.shape[0]-1:
        neighbors.append((point[0]+1,point[1]))
    if point[1]>0:
        neighbors.append((point[0],point[1]-1))
    if point[1]<map.shape[1]-1:
        neighbors.append((point[0],point[1]+1))
    return neighbors

def findPointsAvailable(map_border, start, end):
    # find closest available point to start and end using BFS
    start = np.array(start)
    end = np.array(end)
    explored = np.zeros(map_border.shape).astype(bool)
    queue = [start]
    while len(queue)>0:
        current = queue.pop(0)
        if explored[current[0]][current[1]]:
            continue
        if map_border[current[0]][current[1]] == False:
            start = current
            break
        explored[current[0]][current[1]] = True
        for neighbor in neighbors(map_border, current):
            if not explored[neighbor[0]][neighbor[1]]:
                queue.append(neighbor)
    explored = np.zeros(map_border.shape).astype(bool)
    queue = [end]
    while len(queue)>0:
        current = queue.pop(0)
        if explored[current[0]][current[1]]:
            continue
        if map_border[current[0]][current[1]] == False:
            end = current
            break
        explored[current[0]][current[1]] = True
        for neighbor in neighbors(map_border, current):
            if not explored[neighbor[0]][neighbor[1]]:
                queue.append(neighbor)
    return start, end


def pathfinder(map:np.ndarray, start:np.ndarray, end:np.ndarray, robot_radius=robot_radius):
    """
    Args:
        map: 2D numpy array with 0=free, 1=obstacle
        start: tuple of start coordinates
        end: tuple of end coordinates
    """

    tb = time.time()
    map_border = border_from_map_np(map, robot_radius)
    print(f"Border map generated in {time.time()-tb:.6f}s")
    
    start,end = findPointsAvailable(map_border, start, end)

    grid = np.ones(map.shape).astype(np.float32)
    grid[map_border == True] = np.inf

    assert grid.min() == 1, "cost of moving must be at least 1"

    t0 = time.time()
    path = pyastar2d.astar_path(grid, start, end, allow_diagonal=False)
    dur = time.time() - t0
    if path is None:
        print("No path found")
        return None
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

