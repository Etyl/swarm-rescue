import numpy as np
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
import time
import pyastar2d
import cython

cimport numpy as cnp
from libc.math cimport sqrt

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

SUB_SEGMENT_SIZE = 10 # the number of segment to divide each path segment into
PATH_REFINEMENTS = 3 # number of times to refine the path
save_images = False
debug_mode = False
output = "./solve"


cdef border_from_map_np(map : np.ndarray, robot_radius : int):
    """
    Args:
        map: 2D numpy array of [0,1]U{2} -> 0=free, 1=partially occupied and 2=occupied
        robot_radius: radius of the robot in pixels
    Returns:
        new_map: 2D numpy array of R+ -> 0=free, inf=occupied
    """
    
    new_map = np.zeros(map.shape).astype(np.float32)
    new_map[map > 1.5] = 1
    for _ in range(robot_radius):
        roll_map = (np.roll(new_map,(1, 0), axis=(1, 0))+
                    np.roll(new_map,(0, 1), axis=(1, 0))+
                    np.roll(new_map,(-1, 0), axis=(1, 0))+
                    np.roll(new_map,(0, -1), axis=(1, 0))+
                    np.roll(new_map,(1, 1), axis=(1, 0))+
                    np.roll(new_map,(-1, -1), axis=(1, 0))+
                    np.roll(new_map,(1, -1), axis=(1, 0))+
                    np.roll(new_map,(-1, 1), axis=(1, 0)))
        roll_map[roll_map>0.5] = 1
        new_map += roll_map

    new_map[map > 1.5] = np.inf
    new_map = new_map*robot_radius
    
    bump_map = (len(map)+len(map[0]))*np.ones(map.shape).astype(np.float32)
    bump_map[new_map < 0.5] = 0
    new_map += bump_map

    new_map += np.ones(map.shape)

    return new_map

cdef interpolate_path(cnp.ndarray[cnp.int64_t, ndim=1] point1, cnp.ndarray[cnp.int64_t, ndim=1] point2, float t):
    return point1 + (point2-point1)*t

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative indexing for entire function
@cython.cdivision(True)
cdef is_path_free(cnp.ndarray[cnp.float32_t, ndim=2] map, cnp.ndarray[cnp.int64_t, ndim=1] point1, cnp.ndarray[cnp.int64_t, ndim=1] point2):
    cdef int n = int(np.sum(np.abs(point2-point1)))
    cdef int[2] b = point1
    cdef int[2] a = point2-point1
    cdef int x1,y1,x2,y2
    x1 = b[0]
    y1 = b[1]
    for t in range(1,n+2):
        x2 = b[0] + (a[0]*t)/(n+1)
        y2 = b[1] + (a[1]*t)/(n+1)
        if map[x2][y2] > map[x1][y1]:
            return False
        x1 = x2
        y1 = y2
    return True

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative indexing for entire function
cdef smooth_path(cnp.ndarray[cnp.float32_t, ndim=2] map, cnp.ndarray[cnp.int64_t, ndim=2] path):

    if len(path)<2:
        return path
    cdef int i_ref = 0
    cdef int j_ref = 2
    cdef cnp.ndarray[cnp.int64_t, ndim=2] new_path = np.zeros((len(path),2)).astype(np.int64)
    new_path[0] = path[0]
    cdef int path_index = 1
    cdef int maxpath_index = len(path)
    while i_ref<maxpath_index:
        if j_ref<maxpath_index and is_path_free(map, path[i_ref], path[j_ref]):
            j_ref += 1
        elif j_ref>=maxpath_index:
            new_path[path_index] = path[maxpath_index-1]
            path_index += 1
            break
        else:
            new_path[path_index] = path[j_ref-1]
            path_index += 1
            i_ref = j_ref-1
            j_ref = i_ref+2
    if path_index >= maxpath_index:
        return new_path
    if new_path[max(path_index,maxpath_index-1)][0]!=path[maxpath_index-1][0] or new_path[max(path_index,maxpath_index-1)][1]!=path[maxpath_index-1][1]:
        new_path[path_index] = path[maxpath_index-1]
        path_index += 1
    return new_path[:path_index-1]

@cython.boundscheck(False) # turn off bounds-checking for entire function
@cython.wraparound(False) # turn off negative indexing for entire function
cdef segmentize(cnp.ndarray[cnp.int64_t, ndim=2] path, int i):
    cdef int num_sub_segments = np.sqrt(np.sum((path[i+1]-path[i])**2))/SUB_SEGMENT_SIZE
    return [np.rint(interpolate_path(path[i], path[i+1], t/(num_sub_segments+1))) for t in range(num_sub_segments+2)]

cdef segmentize_path(cnp.ndarray[cnp.float32_t, ndim=2] map, cnp.ndarray[cnp.int64_t, ndim=2] path):
    if len(path)<2:
        return path
    new_path = []
    currentSegment = []
    nextSegment = segmentize(path, 0)
    start_point, end_point, max_segment = 0, 0, np.inf
    for k in range(len(path)-1):
        currentSegment = nextSegment[end_point:]
        if k<len(path)-2:
            nextSegment = segmentize(path, k+1)
        start_point, end_point, max_segment = 0, 0, np.inf
        for i in range(len(currentSegment)-1):
            for j in range(len(nextSegment)-1,-1,-1):
                if is_path_free(map, np.array(currentSegment[i], dtype=DTYPE), np.array(nextSegment[j],dtype=DTYPE)):
                    val = (sqrt(np.sum((currentSegment[i]-currentSegment[0])**2))+
                            sqrt(np.sum((nextSegment[j]-currentSegment[i])**2)) +
                            sqrt(np.sum((nextSegment[len(nextSegment)-1]-nextSegment[j])**2)))
                           
                    if val<max_segment:
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

def findPointsAvailable(map_border : np.ndarray, start, end):
    # find closest available point to start and end using BFS
    start = np.array(start)
    end = np.array(end)
    explored = np.zeros(map_border.shape).astype(bool)
    queue = [start]
    while len(queue)>0:
        current = queue.pop(0)
        if explored[current[0]][current[1]]:
            continue
        if map_border[current[0]][current[1]] < 2*len(map_border)*len(map_border[0]):
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
        if map_border[current[0]][current[1]] < 2*len(map_border)*len(map_border[0]):
            end = current
            break
        explored[current[0]][current[1]] = True
        for neighbor in neighbors(map_border, current):
            if not explored[neighbor[0]][neighbor[1]]:
                queue.append(neighbor)
    if debug_mode:
        print(f"Start : {start}, End : {end}")
        print(f"Start cost : {map_border[start[0]][start[1]]}, End cost : {map_border[end[0]][end[1]]}")
    return start, end


def pathfinder(map:np.ndarray, start:np.ndarray, end:np.ndarray, robot_radius=30):
    """
    Args:
        map: 2D numpy array of [0,1]U{2} -> 0=free, 1=partially occupied and 2=occupied
        start: tuple of start coordinates
        end: tuple of end coordinates
    """
    if debug_mode:

        t0 = time.perf_counter()
        map_border = border_from_map_np(map, robot_radius)
        
        start,end = findPointsAvailable(map_border, start, end)

        # assert map_border.min() > 0.5, "cost of moving must be at least 1"

        tp0 = time.perf_counter()
        path = pyastar2d.astar_path(map_border, start, end, allow_diagonal=False)
        tp = time.perf_counter() - tp0
        if path is None:
            print("No path found")
            return None

        ts0 = time.perf_counter()
        path_smooth = smooth_path(map_border, path.astype(DTYPE))
        
        path_refined = segmentize_path(map_border, np.array(path_smooth).astype(DTYPE))
        path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))
        
        for _ in range(PATH_REFINEMENTS-1):
            path_refined = segmentize_path(map_border, np.array(path_refined).astype(DTYPE))
            path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))
        
        ts = time.perf_counter() - ts0
        
        total_time = time.perf_counter() - t0
        print(f"=========PATHFINDING===========")
        print(f"Time : {total_time:.6f}")
        print(f"A* : {tp/total_time*100:.1f}% | Refinement : {ts/total_time*100:.1f}%")   
    else:    
        map_border = border_from_map_np(map, robot_radius)
        
        start,end = findPointsAvailable(map_border, start, end)

        path = pyastar2d.astar_path(map_border, start, end, allow_diagonal=False)
        if path is None:
            return None

        path_smooth = smooth_path(map_border, path.astype(DTYPE))
        
        path_refined = segmentize_path(map_border, np.array(path_smooth).astype(DTYPE))
        path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))
        
        for _ in range(PATH_REFINEMENTS-1):
            path_refined = segmentize_path(map_border, np.array(path_refined).astype(DTYPE))
            path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))
       
    if save_images:
        img_border = map_border.astype(np.float32).copy()
        plt.imshow(img_border)
        plt.savefig("./border.png")

        current_output = output+"-raw.png"
        plt.figure()
        path_plot = np.array(path)
        plt.plot(path_plot[:,1],path_plot[:,0],color="red")
        plt.imshow(img_border)
        print(f"Plotting path to {current_output}, {len(path)} points)")
        plt.savefig(current_output)

        current_output = output+"-smooth.png"
        plt.figure()
        path_plot = np.array(path_smooth)
        plt.plot(path_plot[:,1],path_plot[:,0],color="red")
        plt.imshow(img_border)
        print(f"Plotting path to {current_output}, {len(path_smooth)} points)")
        plt.savefig(current_output)

        current_output = output+"-refined.png"
        plt.figure()
        path_plot = np.array(path_refined)
        plt.plot(path_plot[:,1],path_plot[:,0],color="red")
        plt.imshow(img_border)
        print(f"Plotting path to {current_output}, {len(path_refined)} points)")
        plt.savefig(current_output)

    
    return path_refined

