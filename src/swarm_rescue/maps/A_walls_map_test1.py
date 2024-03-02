
from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (1113, 750)
# Dimension factor : 1.0
def add_boxes(playground):
    # box 0
    box = NormalBox(up_left_point=(-556, -195),
                    width=180, height=180)
    playground.add(box, box.wall_coordinates)

    # box 1
    box = NormalBox(up_left_point=(-30, 253),
                    width=164, height=104)
    playground.add(box, box.wall_coordinates)


def add_walls(playground):
    # vertical wall 0
    wall = NormalWall(pos_start=(-548, 365),
                      pos_end=(-548, -201))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 1
    wall = NormalWall(pos_start=(-548, 362),
                      pos_end=(338, 362))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 2
    wall = NormalWall(pos_start=(330, 361),
                      pos_end=(550, 361))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 3
    wall = NormalWall(pos_start=(546, 363),
                      pos_end=(546, -366))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 4
    wall = NormalWall(pos_start=(336, 363),
                      pos_end=(336, 255))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 5
    wall = NormalWall(pos_start=(338, 363),
                      pos_end=(338, 255))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(-27, 254),
                      pos_end=(-27, 151))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 7
    wall = NormalWall(pos_start=(-26, 251),
                      pos_end=(133, 251))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 8
    wall = NormalWall(pos_start=(128, 253),
                      pos_end=(128, 151))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 9
    wall = NormalWall(pos_start=(-550, 190),
                      pos_end=(-278, 190))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 10
    wall = NormalWall(pos_start=(-550, 188),
                      pos_end=(-278, 188))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 11
    wall = NormalWall(pos_start=(-427, 191),
                      pos_end=(-427, 154))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 12
    wall = NormalWall(pos_start=(-424, 190),
                      pos_end=(-424, 154))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 13
    wall = NormalWall(pos_start=(-30, 155),
                      pos_end=(132, 155))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 14
    wall = NormalWall(pos_start=(335, 143),
                      pos_end=(335, -112))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 15
    wall = NormalWall(pos_start=(338, 143),
                      pos_end=(338, -112))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 16
    wall = NormalWall(pos_start=(-313, 86),
                      pos_end=(-313, -33))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 17
    wall = NormalWall(pos_start=(-310, 86),
                      pos_end=(-310, -33))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 18
    wall = NormalWall(pos_start=(-550, -29),
                      pos_end=(-223, -29))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 19
    wall = NormalWall(pos_start=(-64, -29),
                      pos_end=(338, -29))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 20
    wall = NormalWall(pos_start=(-550, -31),
                      pos_end=(-224, -31))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 21
    wall = NormalWall(pos_start=(-62, -31),
                      pos_end=(338, -31))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 22
    wall = NormalWall(pos_start=(-62, -106),
                      pos_end=(550, -106))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 23
    wall = NormalWall(pos_start=(-62, -108),
                      pos_end=(550, -108))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 24
    wall = NormalWall(pos_start=(-550, -197),
                      pos_end=(-378, -197))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 25
    wall = NormalWall(pos_start=(-382, -195),
                      pos_end=(-382, -365))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 26
    wall = NormalWall(pos_start=(-384, -362),
                      pos_end=(550, -362))
    playground.add(wall, wall.wall_coordinates)

