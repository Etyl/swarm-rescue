
from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (628, 432)
# Dimension factor : 1.0
def add_boxes(playground):
    pass


def add_walls(playground):
    # horizontal wall 0
    wall = NormalWall(pos_start=(-318, 209),
                      pos_end=(312, 209))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 1
    wall = NormalWall(pos_start=(308, 211),
                      pos_end=(308, -219))
    playground.add(wall, wall.wall_coordinates)

