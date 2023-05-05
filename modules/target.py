import math
import numpy as np
from modules.NewEKF import ExtendedKalmanFilter
from modules.tools import shortest_angular_distance
from modules.armor_detection import Armor


class NormalRobot():
    '''步兵、英雄、工程、哨兵(4装甲板)'''
    def __init__(self) -> None:
        self.target_type = "NormalRobot"
        self.armor = None
        self.armor_id = ""
        self.initial_r = 0.2 # (m)
        self.min_r = 0.2 # (m)
        self.max_r = 0.4 # (m)
        self.max_y_diff = 0.1 * 1.2 # 根据官方机器人制作规范，装甲板真实y坐标的最大可能差值(m)
        
        self.target_state = np.zeros((9,))

        self.last_yaw = 0
        self.last_y = 0
        self.last_r = 0

        # EKF        
        self.q_v = [1e-2, 1e-2, 1e-2, 2e-2, 1, 5e-1, 7e-1, 4e-2, 1e-3]
        self.Q = np.diag(self.q_v)

        self.r_v = [1e-1, 1e-1, 1e-1, 6e-1]
        self.R = np.diag(self.r_v)
        
        self.P0 = np.eye(9)
        
        # xa = x_armor, xc = x_robot_center
        # state: xc, yc, zc, yaw, v_xc, v_yc, v_zc, v_yaw, r
        # measurement: xa, ya, za, yaw
        self.ekfilter = ExtendedKalmanFilter(self.f, self.h, self.j_f, self.j_h, self.Q, self.R, self.P0)

    def init(self, a: Armor) -> None:
        self.armor = a
        self.armor_id = a.name
        xa, ya, za = np.reshape(a.in_imuM, (3,))
        yaw = self.get_continous_yaw(a.yawR_in_imu)

        # Set initial position at r behind the target
        r = self.initial_r
        xc = xa + r * math.sin(yaw)
        yc = ya
        zc = za + r * math.cos(yaw)

        self.target_state = np.zeros((9,))
        self.target_state[0] = xc
        self.target_state[1] = yc
        self.target_state[2] = zc
        self.target_state[3] = yaw
        self.target_state[8] = r

        self.last_y = yc
        self.last_r = r
        self.last_yaw = 0

        self.ekfilter.setState(self.target_state)

        print(self.target_type + " --- Init EKF!")

    def forwardPredict(self, dt):
        ekf_prediction = self.ekfilter.predict(dt)
        return ekf_prediction
    
    def setTargetState(self, state):
        self.target_state = state

    def update(self, matched_armor: Armor) -> None:
        p = np.reshape(matched_armor.in_imuM, (3,))
        measured_yaw = self.get_continous_yaw(matched_armor.yawR_in_imu)
        z = np.array([p[0], p[1], p[2], measured_yaw])
        self.target_state = self.ekfilter.update(z)

    def getPreShotPtsInImu() -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(无重力补偿)'''
        return None
    
    def getArmorPositionFromState(self, x):
        return self.h(x)[:3]
    
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
    
    def get_continous_yaw(self, yaw):
        yaw = self.last_yaw + shortest_angular_distance(self.last_yaw, yaw)
        self.last_yaw = yaw
        return yaw

    def handleArmorJump(self, a: Armor):
        last_yaw = self.target_state[3]
        yaw = self.get_continous_yaw(a.yawR_in_imu)

        if abs(yaw - last_yaw) > 0.4:
            print("Armor jump!")
            self.arrmor_jump = 1
            self.last_y = self.target_state[1]
            self.target_state[1] = np.reshape(a.in_imuM, (3,))[1]
            self.target_state[3] = yaw
            self.target_state[8], self.last_r = self.last_r, self.target_state[8]

        current_p = np.reshape(a.in_imuM, (3,))
        infer_p = self.getArmorPositionFromState(self.target_state)

        if np.linalg.norm(current_p - infer_p) > self.max_match_distance:
            print("State wrong!")
            self.state_error = 1
            r = self.target_state[8]
            self.target_state[0] = current_p[0] + r * math.sin(yaw)
            self.target_state[2] = current_p[2] + r * math.cos(yaw)
            self.target_state[4] = 0
            self.target_state[5] = 0
            self.target_state[6] = 0

        self.ekfilter.setState(self.target_state)

    def limitStateValue(self):
         # Suppress R from converging to zero
        if self.target_state[8] < self.min_r:
            self.target_state[8] = self.min_r
            self.ekfilter.setState(self.target_state)
        elif self.target_state[8] > self.max_r:
            self.target_state[8] = self.max_r
            self.ekfilter.setState(self.target_state)

        if (self.last_y - self.target_state[1]) > self.max_y_diff:
            print("y - error!!")

class BalanceInfantry(NormalRobot):
    '''平衡步兵(2装甲板)'''

class Outpost(NormalRobot):
    '''前哨站(3装甲板)'''

class Base(NormalRobot):
    '''基地(单个静止装甲板)'''

    
