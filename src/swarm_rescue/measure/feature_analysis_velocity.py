import matplotlib.pyplot as plt
import numpy as np

from solutions.utils.types import Vector2D

A = 11.72052117358362
t0 = 20

speed = lambda t: A*(-np.exp(-t/t0)+1)
accel = lambda t: (A-speed(t))/t0
accel_pred = lambda v: (A-v)/t0
command = {"forward": 0.0,
           "lateral": 0.0,
           "rotation": 0.0,
           "grasper": 0}

def get_accel(velocity: Vector2D, control, drone_angle: float) -> Vector2D:
    B = Vector2D(control["forward"], control["lateral"]).rotate(drone_angle)
    if B.norm() > 1:
        B = B.normalize()
    B = A * B
    return Vector2D((B.x - velocity.x)/t0, (B.y - velocity.y)/t0)



def main():
    true_positions = []
    true_velocity = []
    with open("data/raw_measures.txt", "r") as f:
        for line in f:
            l = list(map(float,line.strip().split()))
            true_positions.append(l[:2])
            true_velocity.append(l[2:4])

    true_positions = np.array(true_positions)
    true_velocity = np.array(true_velocity)

    pos_pred = [true_positions[0]]
    vel_pred = [np.array([0,0])]
    for i in range(len(true_velocity)):
        y = pos_pred[-1]
        v = vel_pred[-1] + accel_pred(vel_pred[-1])
        pos_pred.append(y+v)
        vel_pred.append(v)
    pos_pred = np.array(pos_pred)

    plt.figure()
    plt.title("X position")
    plt.plot(range(len(true_positions)), true_positions[:,0], color="blue")
    plt.plot(range(len(pos_pred)), pos_pred[:,0], color="red", linestyle="dashed")


    plt.figure()
    plt.title("X speed")
    plt.plot(range(len(true_velocity)), true_velocity[:,0], color="blue")
    plt.plot(range(len(vel_pred)), vel_pred, color="red", linestyle="dashed")
    #
    #
    # plt.figure()
    # plt.title("Y acceleration")
    # plt.plot(range(len(acceleration)), acceleration[:], color="blue")
    # plt.plot(range(len(acceleration)), accel_pred(velocity[1:-1]), color="red")

    plt.show()


if __name__ == "__main__":
    main()