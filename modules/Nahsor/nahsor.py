import math
from collections import deque

import cv2
import numpy as np

from modules import tools
from modules.autoaim.transformation import LazyTransformation
from modules.tools import shortest_angular_distance


class Nahsor:
    '''能量机关'''

    def __init__(self) -> None:
        self.target_type = "Nahsor"
        self.LazyTrans = None
        self.initial_r = 0.2  # (m)
        self.min_r = 0.2  # (m)
        self.max_r = 0.4  # (m)
        self.max_y_diff = 0.1 * 1.2  # 根据官方机器人制作规范，装甲板真实y坐标的最大可能差值(m)

        self.target_state = np.zeros((9,))

        self.last_yaw = 0
        self.last_y = 0
        self.last_r = 0

        self.state_error = 0

        # EKF
        self.q_v = [1e-2, 1e-2, 1e-2, 2e-2, 1, 5e-1, 7e-1, 4e-2, 1e-3]
        self.Q = np.diag(self.q_v)

        self.r_v = [1e-1, 1e-1, 1e-1, 6e-1]
        self.R = np.diag(self.r_v)

        self.P0 = np.eye(9)

        # xa = x_armor, xc = x_robot_center
        # state: xc, yc, zc, yaw, v_xc, v_yc, v_zc, v_yaw, r
        # measurement: xa, ya, za, yaw

        self.targets_in_pixel = deque(maxlen=3)
        self.shot_point_in_pixel = None
        self.shot_point_in_imu = None

    def init(self, LazyTrans: LazyTransformation) -> None:
        self.LazyTrans = LazyTrans
        xa, ya, za = np.reshape(LazyTrans.in_imu_m, (3,))
        yaw = self.get_continous_yaw(LazyTrans.yaw_in_imu_rad)

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

        print(self.target_type + " --- Init EKF!")

    def setTargetState(self, state):
        self.target_state = state

    def getPredictTime(self, deltatime, bulletSpeed):
        '''获取预测时间'''
        if self.target_state != np.zeros((9,)):
            state = self.target_state
            flyTime = tools.getParaTime(state[:3] * 1000, bulletSpeed) / 1000
            return deltatime + flyTime
        else:  # 第一帧
            # 6439为高台到能量机关距离，2296为r标高度，945为高台高度，200和150为具体位置的修正
            # 假设正前方为能量机关
            state = [6439 + 200, 0, 2296 - 945 - 250]  # 假设的r标位置，仅用于第一帧，应该随意定一个位置也行
            flyTime = tools.getParaTime(state, bulletSpeed) / 1000
            return deltatime + flyTime

    def getPreShotPtsInImu(self, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs,
                           yaw=0, pitch=0) -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        state = self.target_state

        # 后面只改了名，没改内容
        target_0 = np.array(self.getArmorPositionFromState(state)).reshape(3, 1) * 1000  # x y z

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] + math.pi / 2
        _state[8] = self.last_r
        target_1 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] - math.pi / 2
        _state[8] = self.last_r
        target_2 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        # _state = state.copy()
        # _state[1] = self.last_y
        # _state[3] = state[3] + math.pi
        # _state[8] = self.last_r
        # pre_armor_3 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        four_predict_points = [target_0, target_1, target_2]
        # print("aaaa{}".format(self.four_predict_points))

        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(yaw, pitch).T
        R_gimbal2camera = R_camera2gimbal.T

        # 得到三个可疑点的重投影点 armors_in_pixel
        # 与枪管的夹角
        min_angle = 180

        for target_state in four_predict_points:

            # 调试用
            target2_in_imu = target_state
            target2_in_gimbal = R_imu2gimbal @ target2_in_imu
            target2_in_camera = R_gimbal2camera @ target2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            target2_in_pixel, _ = cv2.projectPoints(target2_in_camera, np.zeros((3, 1)), np.zeros((3, 1)), cameraMatrix,
                                                   distCoeffs)
            target2_in_pixel = target2_in_pixel[0][0]
            self.targets_in_pixel.append(target2_in_pixel)

            # 注意单位，单位为mm
            a = (target_state[0] ** 2 + target_state[2] ** 2)
            a = a[0]
            a = math.sqrt(a)
            b = math.sqrt((state[0] * 1000) ** 2 + (state[2] * 1000) ** 2)
            c = self.last_r * 1000

            if tools.is_triangle(a, b, c):
                angle = tools.triangle_angles(a, b, c)

                if angle < min_angle:
                    min_angle = angle
                    self.shot_point_in_pixel = target2_in_pixel
                    self.shot_point_in_imu = target_state

            else:
                min_angle = 0
                self.shot_point_in_pixel = target2_in_pixel
                self.shot_point_in_imu = target_state

        return self.shot_point_in_imu

    def getArmorPositionFromState(self, x):
        return self.h(x)[:3]


    # def j_f(self, x, dt):
    #     # J_f - Jacobian of process function
    #     dfdx = np.eye(9, 9)
    #     dfdx[0, 4] = dt
    #     dfdx[1, 5] = dt
    #     dfdx[2, 6] = dt
    #     dfdx[3, 7] = dt
    #     return dfdx

    def h(self, x):
        # h - Observation function
        z = np.zeros(4)
        xc, yc, zc, yaw, r = x[0], x[1], x[2], x[3], x[8]
        z[0] = xc - r * math.sin(yaw)  # xa
        z[1] = yc  # ya
        z[2] = zc - r * math.cos(yaw)  # za
        z[3] = yaw  # yaw
        return z

    # def j_h(self, x):
    #     # J_h - Jacobian of observation function
    #     dhdx = np.zeros((4, 9))
    #     yaw, r = x[3], x[8]
    #     dhdx[0, 0] = dhdx[1, 1] = dhdx[2, 2] = dhdx[3, 3] = 1
    #     dhdx[0, 3] = -r * math.cos(yaw)
    #     dhdx[2, 3] = r * math.sin(yaw)
    #     dhdx[0, 8] = -math.sin(yaw)
    #     dhdx[2, 8] = -math.cos(yaw)
    #     return dhdx

    def get_continous_yaw(self, yaw):
        yaw = self.last_yaw + shortest_angular_distance(self.last_yaw, yaw)
        self.last_yaw = yaw
        return yaw
    #
    # def handleArmorJump(self, a: LazyTransformation, max_match_distance):
    #     last_yaw = self.target_state[3]
    #     yaw = self.get_continous_yaw(a.yaw_in_imu_rad)
    #
    #     if abs(yaw - last_yaw) > 0.4:
    #         print("Armor jump!")
    #         self.arrmor_jump = 1
    #         self.last_y = self.target_state[1]
    #         self.target_state[1] = np.reshape(a.in_imu_m, (3,))[1]
    #         self.target_state[3] = yaw
    #         self.target_state[8], self.last_r = self.last_r, self.target_state[8]
    #
    #     current_p = np.reshape(a.in_imu_m, (3,))
    #     infer_p = self.getArmorPositionFromState(self.target_state)
    #
    #     if np.linalg.norm(current_p - infer_p) > max_match_distance:
    #         print("State wrong!")
    #         self.state_error = 1
    #         r = self.target_state[8]
    #         self.target_state[0] = current_p[0] + r * math.sin(yaw)
    #         self.target_state[2] = current_p[2] + r * math.cos(yaw)
    #         self.target_state[4] = 0
    #         self.target_state[5] = 0
    #         self.target_state[6] = 0
    #
    #     # self.ekfilter.setState(self.target_state)
    #
    # def limitStateValue(self):
    #     # Suppress R from converging to zero
    #     if self.target_state[8] < self.min_r:
    #         self.target_state[8] = self.min_r
    #         # self.ekfilter.setState(self.target_state)
    #     elif self.target_state[8] > self.max_r:
    #         self.target_state[8] = self.max_r
    #         # self.ekfilter.setState(self.target_state)
    #
    #     if (self.last_y - self.target_state[1]) > self.max_y_diff:
    #         print("y - error!!")
