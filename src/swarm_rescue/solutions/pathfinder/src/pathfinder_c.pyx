import numpy as np
import pyastar2d
import cython
from typing import Optional,Tuple
cimport numpy as cnp
from libc.math cimport sqrt
import matplotlib.pyplot as plt

cnp.import_array()
DTYPE = np.int64
ctypedef cnp.int64_t DTYPE_t

SUB_SEGMENT_SIZE = 10 # the number of segment to divide each path segment into
PATH_REFINEMENTS = 3 # number of times to refine the path


cdef border_from_map_np(map : np.ndarray, robot_radius : int):
    """
    Args:
        map: 2D numpy array of [0,1]U{2} -> 0=free, 1=partially occupied and 2=occupied
        robot_radius: radius of the robot in pixels
    Returns:
        new_map: 2D numpy array of R+ -> 1=free, inf=occupied
    """
    
    new_map = np.zeros_like(map).astype(np.float32)
    roll_map = np.zeros_like(map).astype(np.float32)
    roll_map[map > 1.5] = 1

    for _ in range(1):
        roll_map = (np.roll(roll_map,(1, 0), axis=(1, 0))+
                    np.roll(roll_map,(0, 1), axis=(1, 0))+
                    np.roll(roll_map,(-1, 0), axis=(1, 0))+
                    np.roll(roll_map,(0, -1), axis=(1, 0))+
                    np.roll(roll_map,(1, 1), axis=(1, 0))+
                    np.roll(roll_map,(-1, -1), axis=(1, 0))+
                    np.roll(roll_map,(1, -1), axis=(1, 0))+
                    np.roll(roll_map,(-1, 1), axis=(1, 0)))
        roll_map[roll_map>0.5] = 1
        new_map += roll_map
    hard_borders = np.zeros(roll_map.shape)
    hard_borders[roll_map >= 1] = np.inf

    for _ in range(robot_radius-1):
        roll_map = (np.roll(roll_map,(1, 0), axis=(1, 0))+
                    np.roll(roll_map,(0, 1), axis=(1, 0))+
                    np.roll(roll_map,(-1, 0), axis=(1, 0))+
                    np.roll(roll_map,(0, -1), axis=(1, 0))+
                    np.roll(roll_map,(1, 1), axis=(1, 0))+
                    np.roll(roll_map,(-1, -1), axis=(1, 0))+
                    np.roll(roll_map,(1, -1), axis=(1, 0))+
                    np.roll(roll_map,(-1, 1), axis=(1, 0)))
        roll_map[roll_map>0.5] = 1
        new_map += roll_map
    new_map += hard_borders
    
    bump_map = (len(map)+len(map[0]))*np.ones(map.shape).astype(np.float32)
    bump_map[new_map < 0.5] = 0
    new_map += bump_map

    new_map += np.ones(map.shape)

    return new_map

cdef interpolate_path(cnp.ndarray[cnp.int64_t, ndim=1] point1, cnp.ndarray[cnp.int64_t, ndim=1] point2, float t):
    return point1 + (point2-point1)*t



@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
cdef is_path_free_low(cnp.ndarray[cnp.float32_t, ndim=2] map, int x0,int y0,int x1,int y1):
    cdef int dx = x1 - x0
    cdef int dy = y1 - y0
    cdef int yi = 1
    if dy < 0:
        yi = -1
        dy = -dy
    cdef int D = (2 * dy) - dx
    cdef int y = y0
    cdef int last_x=x0
    cdef int last_y=y0

    for x in range(x0,x1+1):
        if map[x,y] > map[last_x,last_y]:
            return False
        last_x = x
        last_y = y
        if D > 0:
            y = y + yi
            D = D + (2 * (dy - dx))
        else:
            D = D + 2*dy

    return True

@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
cdef is_path_free_high(cnp.ndarray[cnp.float32_t, ndim=2] map, int x0,int y0,int x1,int y1):
    cdef int dx = x1 - x0
    cdef int dy = y1 - y0
    cdef int xi = 1
    if dx < 0:
        xi = -1
        dx = -dx
    cdef int D = (2 * dx) - dy
    cdef int x = x0
    cdef int last_x=x0, last_y=y0

    for y in range(y0, y1+1):
        if map[x,y] > map[last_x,last_y]:
            return False
        last_x = x
        last_y = y
        if D > 0:
            x = x + xi
            D = D + (2 * (dx - dy))
        else:
            D = D + 2*dx

    return True


@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
cdef is_path_free(cnp.ndarray[cnp.float32_t, ndim=2] map, cnp.ndarray[cnp.int64_t, ndim=1] point0, cnp.ndarray[cnp.int64_t, ndim=1] point1):
    cdef int x0=point0[0],y0=point0[1],x1=point1[0],y1=point1[1]
    if abs(y1 - y0) < abs(x1 - x0):
        if x0 > x1:
            return is_path_free_low(map, x1, y1, x0, y0)
        else:
            return is_path_free_low(map, x0, y0, x1, y1)
    else:
        if y0 > y1:
            return is_path_free_high(map, x1, y1, x0, y0)
        else:
            return is_path_free_high(map, x0, y0, x1, y1)


cdef get_max_segment_index(cnp.ndarray[cnp.float32_t, ndim=2] map, cnp.ndarray[cnp.int64_t, ndim=2] path, int i):
    cdef int jmin=i+1, jmax=len(path)-1
    cdef int j = (jmin+jmax)>>1
    while jmax-jmin>1:
        if is_path_free(map, path[i], path[j]):
            jmin = j
        else:
            jmax = j
        j = (jmin + jmax) >> 1
    return j


@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
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
        j_ref = get_max_segment_index(map, path, i_ref)
        if j_ref>=maxpath_index-1:
            new_path[path_index] = path[maxpath_index-1]
            path_index += 1
            break
        else:
            new_path[path_index] = path[j_ref]
            path_index += 1
            i_ref = j_ref

    if path_index >= maxpath_index:
        return new_path
    if new_path[max(path_index,maxpath_index-1)][0]!=path[maxpath_index-1][0] or new_path[max(path_index,maxpath_index-1)][1]!=path[maxpath_index-1][1]:
        new_path[path_index] = path[maxpath_index-1]
        path_index += 1
    return new_path[:path_index-1]

@cython.boundscheck(False) # type: ignore
@cython.wraparound(False) # type: ignore
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

cdef neighbors(map,point):
    return [(point[0]+i,point[1]+j) for i in range(-1,2) for j in range(-1,2)
                 if 0<=point[0]+i<len(map) and 0<=point[1]+j<len(map[0]) and (i,j)!=(0,0)]


cdef findPointsAvailable(map_border : np.ndarray, robot_radius:int, start, end):
    # find closest available point to start and end using BFS
    start = np.array(start)
    end = np.array(end)
    explored = np.zeros(map_border.shape).astype(bool)

    def is_min_local(point):
        for neighbor in neighbors(map_border, point):
            if map_border[neighbor[0]][neighbor[1]] < map_border[point[0]][point[1]]:
                return False
        return True
    
    queue = [start]
    explored[start[0]][start[1]] = True
    while len(queue)>0:
        current = queue.pop(0)
        if map_border[current[0]][current[1]]==1 or (map_border[current[0]][current[1]] <= min(len(map_border)*len(map_border[0])+robot_radius//2,map_border[start[0],start[1]]) and is_min_local(current)):
            start = current
            break
        for neighbor in neighbors(map_border, current):
            if not explored[neighbor[0]][neighbor[1]]:
                queue.append(neighbor)
                explored[neighbor[0]][neighbor[1]] = True

    explored = np.zeros(map_border.shape).astype(bool)
    queue = [end]
    explored[end[0]][end[1]] = True
    while len(queue)>0:
        current = queue.pop(0)
        if map_border[current[0]][current[1]]==1 or (map_border[current[0]][current[1]] < min(len(map_border)*len(map_border[0])+robot_radius//2,map_border[end[0],end[1]]) and is_min_local(current)):
            end = current
            break
        for neighbor in neighbors(map_border, current):
            if not explored[neighbor[0]][neighbor[1]]:
                queue.append(neighbor)
                explored[neighbor[0]][neighbor[1]] = True

    return start, end


cdef find_first_node(map_border, path:np.ndarray):
    cdef int j = 1
    while j<len(path) and is_path_free(map_border, np.array(path[0], dtype=DTYPE), np.array(path[j],dtype=DTYPE)):
        j += 1
    return path[j-1]


def pathfinder(map:np.ndarray, start:np.ndarray, end:np.ndarray, robot_radius) -> Optional[np.ndarray]:
    """
    Params:
        map: 2D numpy array of [0,1]U{2} -> 0=free, 1=partially occupied and 2=occupied
        start: tuple of start coordinates
        end: tuple of end coordinates
    """
    map_border = border_from_map_np(map, robot_radius)

    # img_border = map_border.copy()
    # img_border[img_border==np.inf] = -1
    # img_border[img_border==-1] = 2*np.max(img_border)
    #
    # plt.imsave(f"border.png",img_border)
    # plt.imsave(f"map.png", map)

    start,end = findPointsAvailable(map_border, robot_radius, start, end)

    path = pyastar2d.astar_path(map_border, start, end, allow_diagonal=False)
    if path is None:
        return None

    path_smooth = smooth_path(map_border, path.astype(DTYPE))

    path_refined = segmentize_path(map_border, np.array(path_smooth).astype(DTYPE))
    path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))

    for _ in range(PATH_REFINEMENTS-1):
        path_refined = segmentize_path(map_border, np.array(path_refined).astype(DTYPE))
        path_refined = smooth_path(map_border, np.array(path_refined).astype(DTYPE))

    return path_refined


def pathfinder_fast(map:np.ndarray, start:np.ndarray, end:np.ndarray) -> Tuple[Optional[np.ndarray],Optional[np.ndarray]]:
    """
    Returns the A* path between start and end without any trajectory optimization
    """
    map_border = np.ones(map.shape).astype(np.float32)

    roll_map = np.zeros_like(map).astype(np.float32)
    roll_map[map > 1.5] = 1

    for _ in range(1):
        roll_map = (np.roll(roll_map, (1, 0), axis=(1, 0)) +
                    np.roll(roll_map, (0, 1), axis=(1, 0)) +
                    np.roll(roll_map, (-1, 0), axis=(1, 0)) +
                    np.roll(roll_map, (0, -1), axis=(1, 0)) +
                    np.roll(roll_map, (1, 1), axis=(1, 0)) +
                    np.roll(roll_map, (-1, -1), axis=(1, 0)) +
                    np.roll(roll_map, (1, -1), axis=(1, 0)) +
                    np.roll(roll_map, (-1, 1), axis=(1, 0)))
        roll_map[roll_map > 0.5] = 1
    map_border[roll_map >= 1] = np.inf



    start,end = findPointsAvailable(map_border, 1000, start, end)
    path = pyastar2d.astar_path(map_border, start, end, allow_diagonal=False)

    if path is None:
        return None, None

    first_node = find_first_node(map_border,path)

    return path, first_node
