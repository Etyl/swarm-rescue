TODO

CRITIC
bugfix: drone stops in front of rescue center and rotates on himself - Noé
make drones finish on “return area” when time is close to ending - Noé
reduce bumping (use speed ?)
pathfinding follow path to next pixels
deal with moving wounded
deal with edge case scenarios (eg. window with wounded)
deal with wounded behind kill zone
check features
rescue center is None
deal with kill zone + no com zone
add safe zone (when drone is moving) + merge other safe zones from other drones
detect if drone is not moving inside a kill zone
fix kill zone
fix drone stuck with no path

IMPORTANT
use position confidence when calculating map confidence
map segmentation
chokepoint detection https://ieeexplore.ieee.org/document/7440512
drone communication which area they explore to better spread drones
do launcher for testing
update requirements.txt
post processing map (denoise)
improve localization/mapping
Localization class
test new maps
wounded sharing
merkle tree for confidence map sharing
share drone positions for no GPS zone (TESTED BUT NOT CONCLUSIVE => too much error)
confidence for own position
adapt position for own pos given from other drones according to confidence
wounded physics stop it from being returned (drones have difficulty turning)
cache frontier explorer according to confidence map
get frontier path angle
mapping slim walls (especially doors)
améliorer l'asservissement

NEXT
create RL env (GymEnv) (https://github.com/minhpham160603/SwarmRL)
optimize pathfinder
caching results (segments?)
kd tree
for frontiers: create new function taking list of endpoints
improve map sharing (gossip based + hash confidence map)


current performances:
update_mapping: 38%
find_next_unexplored_target: 26%
map merge: 7%
*


RL:

https://stable-baselines3.readthedocs.io/en/master/guide/rl_tips.html
https://www.marl-book.com/download/marl-book.pdf


SLAM:

gmapping

https://github.com/xiaofeng419/SLAM-2D-LIDAR-SCAN
https://github.com/yingkunwu/FastSLAM

https://www.researchgate.net/publication/4296925_Adaptive_prior_boosting_technique_for_the_efficient_sample_size_in_FastSLAM/download?_tp=eyJjb250ZXh0Ijp7ImZpcnN0UGFnZSI6Il9kaXJlY3QiLCJwYWdlIjoiX2RpcmVjdCJ9fQ



Exploration:

https://www.sciencedirect.com/science/article/pii/S1474667015402460




COMMENTS FINALE

1st place: team n°7 from ENSTA Paris - The team of Fabien Girard and Louis Maurin wins a check of 8 000 €. The jury appreciated the quality and excellent illustrations of the presentation, as well as the various technical aspects:
The pragmatism of the solutions and the overall robustness of the approach,
Excellent management of difficulty zones and their complex interactions,
Communication-independent management of priorities between agents,
The quality of the mapping and visualizations of operations.

2nd place: team n°29 from Telecom Paris - The team of Hippolyte Verninas, Philippe Telo, Emile Le Gallic, Jacques Sun and Noé Vernier wins a check of 4 000 €. The jury appreciated the quality and excellent illustrations of the presentation, as well as the various technical aspects:
The quality of map processing, in particular the notion of confidence used for map sharing,
Path planning and post-processing optimization,
The quality of local navigation and the management of collisions and priorities.

3rd place: team n°6 from ENSTA Paris - The team of Marc-Antoine Oudotte, Victor Morand and Thomas Crasson wins a check of 3 000 €. The jury appreciated the clarity and quality of the presentation, as well as those various technical aspects:
The originality of the trajectory planning solution based on a visibility graph built on a polygonal approximation of the map,
The method of coordination and communication between agents through the distribution of objectives,
Parallelization of the heaviest calculations to maintain reactivity.

