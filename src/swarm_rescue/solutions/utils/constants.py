FRONTIER_COUNT = 6 # the maximum number of selected frontiers
FRONTIER_FEATURES = 5 # the number of features measured for each frontier
OTHER_FRONTIER_INPUTS = 1 # other observations (current % explored area)

OBSERVATION_SPACE = FRONTIER_COUNT*FRONTIER_FEATURES+OTHER_FRONTIER_INPUTS
ACTION_SPACE = FRONTIER_COUNT