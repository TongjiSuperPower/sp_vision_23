import math
import numpy as np
from enum import Enum
from modules.NewEKF import ExtendedKalmanFilter
from modules.armor_detection import Armor


def f(x, dt):
    # f - Process function
    x_new = x.copy()
    x_new[0] += x[4] * dt
    x_new[1] += x[5] * dt
    x_new[2] += x[6] * dt
    x_new[3] += x[7] * dt
    return x_new


def j_f(x, dt):
    # J_f - Jacobian of process function
    dfdx = np.eye(9, 9)
    dfdx[0, 4] = dt
    dfdx[1, 5] = dt
    dfdx[2, 6] = dt
    dfdx[3, 7] = dt
    return dfdx


def h(x):
    # h - Observation function
    z = np.zeros(4)
    xc, yc, zc, yaw, r = x[0], x[1], x[2], x[3], x[8]
    z[0] = xc - r * math.sin(yaw)  # xa
    z[1] = yc                      # ya
    z[2] = zc - r * math.cos(yaw)  # za
    z[3] = yaw                     # yaw
    return z


def j_h(x):
    # J_h - Jacobian of observation function
    dhdx = np.zeros((4, 9))
    yaw, r = x[3], x[8]
    dhdx[0, 0] = dhdx[1, 1] = dhdx[2, 2] = dhdx[3, 3] = 1
    dhdx[0, 3] = -r * math.cos(yaw)
    dhdx[2, 3] = r * math.sin(yaw)
    dhdx[0, 8] = -math.sin(yaw)
    dhdx[2, 8] = -math.cos(yaw)
    return dhdx


class TrackerState(Enum):
    LOST = 0
    DETECTING = 1
    TRACKING = 2
    TEMP_LOST = 3


class Tracker:
    '''单位采用m、s，弧度'''

    def __init__(self, max_match_distance, tracking_threshold, lost_threshold):
        self.max_match_distance = max_match_distance
        self.tracking_threshold = tracking_threshold
        self.lost_threshold = lost_threshold

        self.tracker_state = TrackerState.LOST

        self.tracked_armor: Armor = None
        self.tracked_id = ""
        self.target_state = np.zeros((9,))

        self.lost_count = 0
        self.detect_count = 0

        self.last_yaw = 0
        self.last_y = 0
        self.last_r = 0

        # EKF参数:
        # Q - process noise covariance matrix
        q_v = [1e-2, 1e-2, 1e-2, 2e-2, 5e-2, 5e-2, 1e-4, 4e-2, 1e-3]
        Q = np.diag(q_v)
        # R
        r_v = [1e-1, 1e-1, 1e-1, 2e-1]
        R = np.diag(r_v)
        # P - error estimate covariance matrix
        P0 = np.eye(9)

        # EKF
        # xa = x_armor, xc = x_robot_center
        # state: xc, yc, zc, yaw, v_xc, v_yc, v_zc, v_yaw, r
        # measurement: xa, ya, za, yaw
        self.ekf = ExtendedKalmanFilter(f, h, j_f, j_h, Q, R, P0)

    def init(self, armors: list[Armor]):
        # Simply choose the armor that is closest to image center
        self.tracked_armor = armors[0]  # armors之前已经根据距离进行了排序，所以[0]就是最近的

        self.initEKF(self.tracked_armor)
        self.tracked_id = self.tracked_armor.name
        self.tracker_state = TrackerState.DETECTING

    def update(self, armors: list[Armor], dt):
        # KF predict
        ekf_prediction = self.ekf.predict(dt)
        print("EKF predict")

        matched = False

        # Use KF prediction as default target state if no matched armor is found
        self.target_state = ekf_prediction

        if len(armors) > 0:
            min_position_diff = float('inf')
            predicted_position = self.getArmorPositionFromState(ekf_prediction)

            for armor in armors:
                position_vec = np.reshape(armor.in_imuM, (3,))

                position_diff = np.linalg.norm(predicted_position - position_vec)

                if position_diff < min_position_diff:
                    min_position_diff = position_diff
                    self.tracked_armor = armor

            if min_position_diff < self.max_match_distance:
                matched = True

                # Update EKF
                p = np.reshape(self.tracked_armor.in_imuM, (3,))
                measured_yaw = self.tracked_armor.yawR_in_imu
                z = np.array([p[0], p[1], p[2], measured_yaw])
                target_state = self.ekf.update(z)
                print("EKF update")
            else:
                # Check if there is same id armor in current frame
                for armor in armors:
                    if armor.name == self.tracked_id:
                        # Armor jump happens
                        matched = True
                        self.tracked_armor = armor
                        self.handleArmorJump(self.tracked_armor)
                        break

        # Suppress R from converging to zero
        if self.target_state[8] < 0.2:
            self.target_state[8] = 0.2
            self.ekf.setState(self.target_state)

        # Tracking state machine
        if self.tracker_state == TrackerState.DETECTING:
            if matched:
                self.detect_count += 1
                if self.detect_count > self.tracking_threshold:
                    self.detect_count = 0
                    self.tracker_state = TrackerState.TRACKING
            else:
                self.detect_count = 0
                self.tracker_state = TrackerState.LOST

        elif self.tracker_state == TrackerState.TRACKING:
            if not matched:
                self.tracker_state = TrackerState.TEMP_LOST
                self.lost_count += 1
                
        elif self.tracker_state == TrackerState.TEMP_LOST:
            if not matched:
                self.lost_count += 1
                if self.lost_count > self.lost_threshold:
                    self.lost_count = 0
                    self.tracker_state = TrackerState.LOST
            else:
                self.tracker_state = TrackerState.TRACKING
                self.lost_count = 0

    def initEKF(self, a: Armor):
        xa, ya, za = np.reshape(a.in_imuM, (3,))
        yaw = a.yawR_in_imu

        r = 0.2
        xc = xa + r * math.sin(yaw)
        yc = ya
        zc = za + r * math.cos(yaw)

        # Set initial position at 0.2m behind the target
        self.target_state = np.zeros((9,))
        self.target_state[0] = xc
        self.target_state[1] = yc
        self.target_state[2] = zc
        self.target_state[3] = yaw
        self.target_state[8] = r

        self.last_y = yc
        self.last_r = r
        self.last_yaw = 0

        self.ekf.setState(self.target_state)

        print("Init EKF!")

    def handleArmorJump(self, a: Armor):
        self.last_yaw = self.target_state[3]
        yaw = a.yawR_in_imu

        if abs(yaw - self.last_yaw) > 0.4:
            print("Armor jump!")
            self.last_y = self.target_state[2]
            self.target_state[2] = np.reshape(a.in_imuM, (3,))[2]
            self.target_state[3] = yaw
            self.target_state[8], self.last_r = self.last_r, self.target_state[8]

        current_p = np.reshape(a.in_imuM, (3,))
        infer_p = self.getArmorPositionFromState(self.target_state)

        if np.linalg.norm(current_p - infer_p) > self.max_match_distance:
            print("State wrong!")
            r = self.target_state[8]
            self.target_state[0] = current_p[0] + r * math.sin(yaw)
            self.target_state[2] = current_p[2] + r * math.cos(yaw)
            self.target_state[4] = 0
            self.target_state[6] = 0

        self.ekf.setState(self.target_state)

    def getArmorPositionFromState(self, x):
        return h(x)[:3]
    