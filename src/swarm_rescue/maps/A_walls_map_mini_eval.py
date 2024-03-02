
from spg_overlay.entities.normal_wall import NormalWall, NormalBox


# Dimension of the map : (1540, 1115)
# Dimension factor : 1.0
def add_boxes(playground):
    # box 0
    box = NormalBox(up_left_point=(-754, -390),
                    width=123, height=88)
    playground.add(box, box.wall_coordinates)

    # box 1
    box = NormalBox(up_left_point=(-554, -378),
                    width=80, height=164)
    playground.add(box, box.wall_coordinates)

    # box 2
    box = NormalBox(up_left_point=(478, 500),
                    width=89, height=148)
    playground.add(box, box.wall_coordinates)

    # box 3
    box = NormalBox(up_left_point=(-68, 546),
                    width=89, height=157)
    playground.add(box, box.wall_coordinates)

    # box 4
    box = NormalBox(up_left_point=(-231, 546),
                    width=88, height=157)
    playground.add(box, box.wall_coordinates)


def add_walls(playground):
    # horizontal wall 0
    wall = NormalWall(pos_start=(-755, 543),
                      pos_end=(750, 543))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 1
    wall = NormalWall(pos_start=(-753, 541),
                      pos_end=(-225, 541))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 2
    wall = NormalWall(pos_start=(-149, 546),
                      pos_end=(-149, 391))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 3
    wall = NormalWall(pos_start=(-151, 542),
                      pos_end=(-62, 542))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 4
    wall = NormalWall(pos_start=(15, 546),
                      pos_end=(15, 391))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 5
    wall = NormalWall(pos_start=(13, 541),
                      pos_end=(748, 541))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 6
    wall = NormalWall(pos_start=(-374, 544),
                      pos_end=(-374, 296))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 7
    wall = NormalWall(pos_start=(-229, 544),
                      pos_end=(-229, 390))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 8
    wall = NormalWall(pos_start=(-231, 394),
                      pos_end=(-146, 394))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 9
    wall = NormalWall(pos_start=(-66, 544),
                      pos_end=(-66, 390))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 10
    wall = NormalWall(pos_start=(-68, 394),
                      pos_end=(18, 394))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 11
    wall = NormalWall(pos_start=(217, 544),
                      pos_end=(217, 204))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 12
    wall = NormalWall(pos_start=(-372, 544),
                      pos_end=(-372, 293))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 13
    wall = NormalWall(pos_start=(-567, 298),
                      pos_end=(-370, 298))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 14
    wall = NormalWall(pos_start=(219, 544),
                      pos_end=(219, 204))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 15
    wall = NormalWall(pos_start=(480, 502),
                      pos_end=(480, 355))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 16
    wall = NormalWall(pos_start=(481, 498),
                      pos_end=(565, 498))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 17
    wall = NormalWall(pos_start=(561, 500),
                      pos_end=(561, 354))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 18
    wall = NormalWall(pos_start=(-754, 420),
                      pos_end=(-609, 420))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 19
    wall = NormalWall(pos_start=(-753, 418),
                      pos_end=(-610, 418))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 20
    wall = NormalWall(pos_start=(478, 358),
                      pos_end=(564, 358))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 21
    wall = NormalWall(pos_start=(-567, 300),
                      pos_end=(-371, 300))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 22
    wall = NormalWall(pos_start=(-752, 548),
                      pos_end=(-752, -542))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 23
    wall = NormalWall(pos_start=(-750, 546),
                      pos_end=(-750, -396))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 24
    wall = NormalWall(pos_start=(-752, -393),
                      pos_end=(-634, -393))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 25
    wall = NormalWall(pos_start=(-637, -390),
                      pos_end=(-637, -476))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 26
    wall = NormalWall(pos_start=(-588, 202),
                      pos_end=(-588, 136))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 27
    wall = NormalWall(pos_start=(-586, 202),
                      pos_end=(-586, 136))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 28
    wall = NormalWall(pos_start=(-366, 202),
                      pos_end=(-366, -374))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 29
    wall = NormalWall(pos_start=(-364, 202),
                      pos_end=(-364, -374))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 30
    wall = NormalWall(pos_start=(745, 544),
                      pos_end=(745, -540))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 31
    wall = NormalWall(pos_start=(747, 546),
                      pos_end=(747, -542))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 32
    wall = NormalWall(pos_start=(-753, 177),
                      pos_end=(-715, 177))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 33
    wall = NormalWall(pos_start=(-389, 177),
                      pos_end=(-363, 177))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 34
    wall = NormalWall(pos_start=(-614, 176),
                      pos_end=(-492, 176))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 35
    wall = NormalWall(pos_start=(-752, 174),
                      pos_end=(-716, 174))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 36
    wall = NormalWall(pos_start=(-614, 174),
                      pos_end=(-493, 174))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 37
    wall = NormalWall(pos_start=(-388, 174),
                      pos_end=(-364, 174))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 38
    wall = NormalWall(pos_start=(579, 113),
                      pos_end=(748, 113))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 39
    wall = NormalWall(pos_start=(-279, 112),
                      pos_end=(493, 112))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 40
    wall = NormalWall(pos_start=(-279, 110),
                      pos_end=(492, 110))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 41
    wall = NormalWall(pos_start=(580, 110),
                      pos_end=(747, 110))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 42
    wall = NormalWall(pos_start=(589, 114),
                      pos_end=(589, -106))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 43
    wall = NormalWall(pos_start=(591, 112),
                      pos_end=(591, -106))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 44
    wall = NormalWall(pos_start=(-588, 48),
                      pos_end=(-588, -79))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 45
    wall = NormalWall(pos_start=(-586, 48),
                      pos_end=(-586, -79))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 46
    wall = NormalWall(pos_start=(-619, -75),
                      pos_end=(-486, -75))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 47
    wall = NormalWall(pos_start=(-280, 18),
                      pos_end=(482, 18))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 48
    wall = NormalWall(pos_start=(-279, 16),
                      pos_end=(483, 16))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 49
    wall = NormalWall(pos_start=(30, 19),
                      pos_end=(30, -7))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 50
    wall = NormalWall(pos_start=(28, 18),
                      pos_end=(28, -8))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 51
    wall = NormalWall(pos_start=(-753, -74),
                      pos_end=(-716, -74))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 52
    wall = NormalWall(pos_start=(-390, -74),
                      pos_end=(-363, -74))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 53
    wall = NormalWall(pos_start=(-752, -76),
                      pos_end=(-717, -76))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 54
    wall = NormalWall(pos_start=(-618, -77),
                      pos_end=(-486, -77))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 55
    wall = NormalWall(pos_start=(-389, -76),
                      pos_end=(-364, -76))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 56
    wall = NormalWall(pos_start=(-532, -74),
                      pos_end=(-532, -295))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 57
    wall = NormalWall(pos_start=(-533, -290),
                      pos_end=(-364, -290))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 58
    wall = NormalWall(pos_start=(-530, -74),
                      pos_end=(-530, -293))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 59
    wall = NormalWall(pos_start=(-532, -288),
                      pos_end=(-364, -288))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 60
    wall = NormalWall(pos_start=(-153, -215),
                      pos_end=(77, -215))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 61
    wall = NormalWall(pos_start=(589, -196),
                      pos_end=(589, -221))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 62
    wall = NormalWall(pos_start=(591, -196),
                      pos_end=(591, -221))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 63
    wall = NormalWall(pos_start=(184, -217),
                      pos_end=(621, -217))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 64
    wall = NormalWall(pos_start=(-279, -214),
                      pos_end=(-242, -214))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 65
    wall = NormalWall(pos_start=(708, -217),
                      pos_end=(748, -217))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 66
    wall = NormalWall(pos_start=(-279, -216),
                      pos_end=(-243, -216))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 67
    wall = NormalWall(pos_start=(28, -106),
                      pos_end=(28, -538))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 68
    wall = NormalWall(pos_start=(30, -106),
                      pos_end=(30, -538))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 69
    wall = NormalWall(pos_start=(-152, -217),
                      pos_end=(77, -217))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 70
    wall = NormalWall(pos_start=(185, -219),
                      pos_end=(621, -219))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 71
    wall = NormalWall(pos_start=(709, -218),
                      pos_end=(747, -218))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 72
    wall = NormalWall(pos_start=(323, -216),
                      pos_end=(323, -404))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 73
    wall = NormalWall(pos_start=(325, -216),
                      pos_end=(325, -404))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 74
    wall = NormalWall(pos_start=(-552, -377),
                      pos_end=(-552, -540))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 75
    wall = NormalWall(pos_start=(-551, -381),
                      pos_end=(-476, -381))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 76
    wall = NormalWall(pos_start=(-480, -378),
                      pos_end=(-480, -539))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 77
    wall = NormalWall(pos_start=(-482, -535),
                      pos_end=(747, -535))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 78
    wall = NormalWall(pos_start=(-754, -473),
                      pos_end=(-634, -473))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 79
    wall = NormalWall(pos_start=(-750, -470),
                      pos_end=(-750, -540))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 80
    wall = NormalWall(pos_start=(-752, -536),
                      pos_end=(-549, -536))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 81
    wall = NormalWall(pos_start=(323, -501),
                      pos_end=(323, -539))
    playground.add(wall, wall.wall_coordinates)

    # vertical wall 82
    wall = NormalWall(pos_start=(325, -501),
                      pos_end=(325, -537))
    playground.add(wall, wall.wall_coordinates)

    # horizontal wall 83
    wall = NormalWall(pos_start=(-754, -537),
                      pos_end=(750, -537))
    playground.add(wall, wall.wall_coordinates)

