
import math
import numpy as np
from spg_overlay.utils.utils import circular_mean, normalize_angle



# Covariance for EKF simulation
Q = np.diag([
    0.1,  # variance of location on x-axis
    0.1,  # variance of location on y-axis
    np.deg2rad(1.0),  # variance of yaw angle
    1.0  # variance of velocity
]) ** 2  # predict state covariance
R = np.diag([1.0, 1.0]) ** 2  # Observation x,y position covariance





def motion_model(x, u):
    F = np.array([[1.0, 0, 0, 0],
                  [0, 1.0, 0, 0],
                  [0, 0, 1.0, 0],
                  [0, 0, 0, 0]])

    B = np.array([[math.cos(x[2, 0]), 0],
                  [math.sin(x[2, 0]), 0],
                  [0.0, 1.0],
                  [1.0, 0.0]])

    x = F @ x + B @ u

    return x


def observation_model(x):
    H = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    z = H @ x

    return z


def jacob_f(x, u):
    """
    Jacobian of Motion Model

    motion model
    x_{t+1} = x_t+v*dt*cos(yaw)
    y_{t+1} = y_t+v*dt*sin(yaw)
    yaw_{t+1} = yaw_t+omega*dt
    v_{t+1} = v{t}
    so
    dx/dyaw = -v*dt*sin(yaw)
    dx/dv = dt*cos(yaw)
    dy/dyaw = v*dt*cos(yaw)
    dy/dv = dt*sin(yaw)
    """
    yaw = x[2, 0]
    v = u[0, 0]
    jF = np.array([
        [1.0, 0.0, -v * math.sin(yaw), math.cos(yaw)],
        [0.0, 1.0, v * math.cos(yaw), math.sin(yaw)],
        [0.0, 0.0, 1.0, 0.0],
        [0.0, 0.0, 0.0, 1.0]])

    return jF


def jacob_h():
    # Jacobian of Observation Model
    jH = np.array([
        [1, 0, 0, 0],
        [0, 1, 0, 0]
    ])

    return jH


def ekf_estimation(xEst, PEst, z, u):
    #  Predict
    xPred = motion_model(xEst, u)
    jF = jacob_f(xEst, u)
    PPred = jF @ PEst @ jF.T + Q

    #  Update
    jH = jacob_h()
    zPred = observation_model(xPred)
    y = z - zPred
    S = jH @ PPred @ jH.T + R
    K = PPred @ jH.T @ np.linalg.inv(S)
    xEst = xPred + K @ y
    PEst = (np.eye(len(xEst)) - K @ jH) @ PPred
    return xEst, PEst





class Localizer():
    def __init__(self):

        # State Vector [x y yaw v]'
        self.xEst = np.zeros((4, 1))
        self.PEst = np.eye(4)

        self.xDR = np.zeros((4, 1))  # Dead reckoning

        # history
        self.hxEst = []
        self.hz = []

        self.isInitialized = False

    def localize(self, gps_pos, odometer_v, odometer_angle):

        if not self.isInitialized:
            self.xEst = np.array([[gps_pos[0]], [gps_pos[1]], [0.0], [0.0]])
            self.isInitialized = True

        def angle(v1, v2):
            return math.atan2(v2[1],v2[0]) - math.atan2(v1[1],v1[0])
        
        prev_angle = self.xEst[2, 0]

        v_angle = angle(odometer_v/np.linalg.norm(odometer_v), [0,1]) + np.pi/4
        v = np.linalg.norm(odometer_v)
        u = np.array([[v], [normalize_angle(v_angle-prev_angle)]])

        if gps_pos is None:
            self.xEst = motion_model(self.xEst, u)
        
        else:
            z = np.array([[gps_pos[0]],[gps_pos[1]]])

            self.xEst, self.PEst = ekf_estimation(self.xEst, self.PEst, z, u)
            self.hz.append((self.hz, z))

        print(u)
        print(self.xEst)

        # store data history
        self.hxEst.append((self.hxEst, self.xEst))
        

        return self.xEst

