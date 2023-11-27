"""
MyMapIntermediate01
"""

# maximum number of times the drone can be in the same position
# i.e the maximum number of times the check_target_reached function can return False
_LOOP_COUNT_GOING_TO_TARGET_THRESHOLD = 100

# maximum number of times the drone can be close to the previous searching start point
# i.e the maximum number of times the test_position_close_start_point function can return True
_COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD = 50

# maximum number of times the drone can't find a target
# i.e the maximum number of times the search_for_target function can return None
_NONE_TARGET_FOUND_THRESHOLD = 10

# the thickness of the walls in the map when the path is computed (in order to allow a smoother path)
_WALL_THICKENING = 4

# the sampling rate of the path (in order to reduce the number of points)
_PATH_SAMPLING_RATE = 1

"""
MyMapIntermediate02
"""

# maximum number of times the drone can be in the same position
# i.e the maximum number of times the check_target_reached function can return False
_LOOP_COUNT_GOING_TO_TARGET_THRESHOLD = 1000

# maximum number of times the drone can be close to the previous searching start point
# i.e the maximum number of times the test_position_close_start_point function can return True
_COUNT_CLOSE_PREVIOUS_SEARCHING_START_POINT_THRESHOLD = 100

# maximum number of times the drone can't find a target
# i.e the maximum number of times the search_for_target function can return None
_NONE_TARGET_FOUND_THRESHOLD = 20

# the thickness of the walls in the map when the path is computed (in order to allow a smoother path)
_WALL_THICKENING = 2

# the sampling rate of the path (in order to reduce the number of points)
_PATH_SAMPLING_RATE = 4
