FRONTIER_COUNT:int = 6 # the maximum number of selected frontiers
FRONTIER_FEATURES:int = 6 # the number of features measured for each frontier
OTHER_FRONTIER_INPUTS:int = 3 # other observations (current % explored area)

REWARD_SPACE = 2
OBSERVATION_SPACE:int = FRONTIER_COUNT*FRONTIER_FEATURES+OTHER_FRONTIER_INPUTS
ACTION_SPACE:int = FRONTIER_COUNT