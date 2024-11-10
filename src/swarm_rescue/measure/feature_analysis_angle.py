import matplotlib.pyplot as plt
import numpy as np

from solutions.utils.types import Vector2D
from solutions.utils.utils import normalize_angle

A = 11.72052117358362
t0 = 20

A2 = 0.171

speed = lambda t: A*(-np.exp(-t/t0)+1)
accel = lambda t: (A-speed(t))/t0
accel_pred = lambda v: (A-v)/t0
command = {"forward": 0.0,
           "lateral": 0.0,
           "rotation": 0.0,
           "grasper": 0}

def get_accel(velocity: Vector2D, control, drone_angle: float) -> Vector2D:
    B = Vector2D(control["forward"], control["lateral"])
    B.rotate(drone_angle)
    if B.norm() > 1:
        B = B.normalize()
    B = A * B
    return Vector2D((B.x - velocity.x)/t0, (B.y - velocity.y)/t0)

def get_angular_velocity(control) -> float:
    return A2 * command["rotation"]

def main():
    true_angle = []
    true_angle_velocity = []
    with open("data/raw_measures.txt", "r") as f:
        for line in f:
            l = list(map(float,line.strip().split()))
            true_angle.append(l[4])
            true_angle_velocity.append(l[5])

    true_angle = np.array(true_angle)
    true_angle_velocity = np.array(true_angle_velocity)

    angle_pred = [true_angle[0]]
    for i in range(len(true_angle_velocity)):
        a = angle_pred[-1]
        w = true_angle_velocity[i]
        angle_pred.append(normalize_angle(a+w))
    angle_pred = np.array(angle_pred)

    plt.figure()
    plt.title("angle")
    plt.plot(range(len(true_angle)), true_angle, color="blue")
    plt.plot(range(len(angle_pred)), angle_pred, color="red", linestyle="dashed")


    plt.figure()
    plt.title("angular velocity")
    plt.plot(range(len(true_angle_velocity)), true_angle_velocity, color="blue")
    # plt.plot(range(len(velocity)), velocity_pred, color="red", linestyle="dashed")
    #
    #
    # plt.figure()
    # plt.title("Y acceleration")
    # plt.plot(range(len(acceleration)), acceleration[:], color="blue")
    # plt.plot(range(len(acceleration)), accel_pred(velocity[1:-1]), color="red")

    print(true_angle_velocity[150])

    plt.show()


if __name__ == "__main__":
    main()