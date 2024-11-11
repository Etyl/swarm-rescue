import matplotlib.pyplot
import numpy as np
from matplotlib import pyplot as plt

from spg_overlay.utils.utils import normalize_angle


def MSE(arr1, arr2):
    return np.linalg.norm(arr1 - arr2)

def MAE(arr1,arr2):
    return np.mean(np.abs(arr1 - arr2))

def angle_distance(a1, a2)-> float:
    a1 = normalize_angle(a1)
    a2 = normalize_angle(a2)
    r = abs(a1 - a2)
    r = min(r, abs(a1-a2+2*np.pi))
    r = min(r, abs(a1-a2-2*np.pi))
    return r

def main():
    true_pos = []
    drone_pos = []
    meas_pos = []
    true_vel = []
    drone_vel = []
    meas_vel = []

    true_angle = []
    drone_angle = []
    meas_angle = []
    with open("./data/measured_pos.txt","r") as f:
        for line in f:
            l = list(map(float, line.strip().split()))
            true_pos.append(l[:2])
            drone_pos.append(l[2:4])
            meas_pos.append(l[4:6])
            true_vel.append(l[6:8])
            drone_vel.append(l[8:10])
            meas_vel.append(l[10:12])
            true_angle.append(l[12])
            drone_angle.append(l[13])
            meas_angle.append(l[14])

    true_pos = np.array(true_pos)
    drone_pos = np.array(drone_pos)
    meas_pos = np.array(meas_pos)
    true_vel = np.array(true_vel)
    drone_vel = np.array(drone_vel)
    meas_vel = np.array(meas_vel)
    true_angle = np.array(true_angle)
    drone_angle = np.array(drone_angle)
    meas_angle = np.array(meas_angle)

    avg_size = 3
    avg_pos = []
    for i in range(avg_size-1):
        avg_pos.append(meas_pos[i])
    for i in range(avg_size-1,len(meas_pos)):
        avg_pos.append(np.mean(meas_pos[i-avg_size+1:i+1],axis=0))
    avg_pos = np.array(avg_pos)

    avg_vel = []
    for i in range(avg_size - 1):
        avg_vel.append(meas_vel[i])
    for i in range(avg_size - 1, len(meas_vel)):
        avg_vel.append(np.mean(meas_vel[i-avg_size+1:i + 1], axis=0))
    avg_vel = np.array(avg_vel)


    plt.figure()
    plt.title("Measure error")
    plt.plot((meas_pos-true_pos)[:,0],color="blue")

    plt.figure()
    plt.title("Measure avg error")
    plt.plot((avg_pos - true_pos)[:, 0], color="blue")

    plt.figure()
    plt.title("drone error")
    plt.plot((drone_pos-true_pos)[:,0],color="red")

    plt.figure()
    plt.title("Measure error vel")
    plt.plot((meas_vel - true_vel)[:, 0], color="blue")
    plt.ylim(-4,4)


    plt.figure()
    plt.title("Measure avg error vel")
    plt.plot((avg_vel - true_vel)[:, 0], color="blue")
    plt.ylim(-4,4)


    plt.figure()
    plt.title("Drone error vel")
    plt.plot((drone_vel - true_vel)[:, 0], color="red")
    plt.ylim(-4,4)

    plt.figure()
    plt.title("Drone error angle")
    plt.plot([angle_distance(x,x2) for x,x2 in zip(drone_angle,true_angle)], color="red")

    plt.figure()
    plt.title("Measure error angle")
    plt.plot([angle_distance(x,x2) for x,x2 in zip(meas_angle,true_angle)], color="red")


    # plt.figure()
    # plt.title("X vel")
    # plt.plot(np.arange(len(true_vel)), true_vel[:, 0], color='blue')
    # plt.plot(np.arange(len(true_vel)), drone_vel[:, 0], color='red', linestyle=':')
    # plt.plot(np.arange(len(true_vel)), meas_vel[:, 0], color='green', linestyle=':')
    #
    # plt.figure()
    # plt.title("Y vel")
    # plt.plot(np.arange(len(true_vel)), true_vel[:, 1], color='blue')
    # plt.plot(np.arange(len(true_vel)), drone_vel[:, 1], color='red', linestyle=':')
    # plt.plot(np.arange(len(true_vel)), meas_vel[:, 1], color='green', linestyle=':')
    #
    # plt.figure()
    # plt.title("X")
    # plt.plot(np.arange(len(true_pos)), true_pos[:,0], color='blue')
    # plt.plot(np.arange(len(true_pos)), drone_pos[:,0], color='red', linestyle=':')
    # plt.plot(np.arange(len(true_pos)), meas_pos[:,0], color='green', linestyle=':')
    #
    # plt.figure()
    # plt.title("Y")
    # plt.plot(np.arange(len(true_pos)), true_pos[:, 1], color='blue')
    # plt.plot(np.arange(len(true_pos)), drone_pos[:, 1], color='red', linestyle=':')
    # plt.plot(np.arange(len(true_pos)), meas_pos[:, 1], color='green', linestyle=':')
    #
    # plt.figure()
    # plt.title("angle")
    # plt.plot(true_angle, color='blue')
    # plt.plot(drone_angle, color='red', linestyle=':')
    # plt.plot(meas_angle, color='green', linestyle=':')



    print("==== ANGLE ====")
    print_error(drone_angle, meas_angle, true_angle)
    print("==== POS ====")
    print_error(drone_pos, meas_pos, true_pos)
    print(f"MSE avg: {np.linalg.norm(true_pos-avg_pos)}")
    print(f"MAE avg: {MAE(true_pos, avg_pos)}")
    print("==== VELOCITY ====")
    print_error(drone_vel, meas_vel, true_vel)
    print(f"MSE avg: {np.linalg.norm(true_vel - avg_vel)}")
    print(f"MAE avg: {MAE(true_vel, avg_vel)}")

    plt.show()


def print_error(drone_arr, meas_arr, true_arr):
    print(f"MSE drone: {np.linalg.norm(true_arr - drone_arr)}")
    print(f"MSE measured: {np.linalg.norm(true_arr - meas_arr)}")
    print(f"MAE drone: {MAE(true_arr, drone_arr)}")
    print(f"MAE measured: {MAE(true_arr, meas_arr)}")


if __name__ == '__main__':
    main()