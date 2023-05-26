import math
import cv2
import numpy as np
import modules.tools as tools
from modules.NewEKF import ExtendedKalmanFilter
from modules.tools import shortest_angular_distance
from modules.autoaim.armor import Armor
from collections import deque


class NormalRobot():
    '''步兵、英雄、工程、哨兵(4装甲板)'''

    def __init__(self) -> None:
        self.target_type = "NormalRobot"
        self.armor = None
        self.armor_id = ""
        self.initial_r = 0.2  # (m)
        self.min_r = 0.2  # (m)
        self.max_r = 0.4  # (m)
        self.max_y_diff = 0.1 * 1.2  # 根据官方机器人制作规范，装甲板真实y坐标的最大可能差值(m)

        self.target_state = np.zeros((9,))
        self.pre_target_state = np.zeros((9,))

        self.last_yaw = 0
        self.last_y = 0
        self.last_r = 0

        self.arrmor_jump = 0
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
        self.ekfilter = ExtendedKalmanFilter(self.f, self.h, self.j_f, self.j_h, self.Q, self.R, self.P0)

        self.armors_in_pixel = deque(maxlen=3)
        self.shot_point_in_pixel = None
        self.shot_point_in_imu = None

    def init(self, a: Armor) -> None:
        self.armor = a
        self.armor_id = a.name
        xa, ya, za = np.reshape(a.in_imu_m, (3,))
        yaw = self.get_continous_yaw(a.yaw_in_imu_rad)

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
        p = np.reshape(matched_armor.in_imu_m, (3,))
        measured_yaw = self.get_continous_yaw(matched_armor.yaw_in_imu_rad)
        z = np.array([p[0], p[1], p[2], measured_yaw])
        self.target_state = self.ekfilter.update(z)

    def getPreShotPtsInImu(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0) -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        state = self.target_state

        flyTime = tools.getParaTime(state[:3] * 1000, bulletSpeed) / 1000

        state = self.f(state, deltatime+flyTime)  # predicted

        pre_armor_0 = np.array(self.getArmorPositionFromState(state)).reshape(3, 1) * 1000  # x y z

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] + math.pi/2
        _state[8] = self.last_r
        pre_armor_1 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] - math.pi/2
        _state[8] = self.last_r
        pre_armor_2 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        # _state = state.copy()
        # _state[1] = self.last_y
        # _state[3] = state[3] + math.pi
        # _state[8] = self.last_r
        # pre_armor_3 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        four_predict_points = [pre_armor_0, pre_armor_1, pre_armor_2]
        # print("aaaa{}".format(self.four_predict_points))

        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(yaw, pitch).T
        R_gimbal2camera = R_camera2gimbal.T

        # 得到三个可疑点的重投影点 armors_in_pixel
        # 与枪管的夹角
        min_angle = 180

        for armor_state in four_predict_points:

            # 调试用
            armor2_in_imu = armor_state
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3, 1)), np.zeros((3, 1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            self.armors_in_pixel.append(armor2_in_pixel)

            # 注意单位，单位为mm
            a = (armor_state[0]**2 + armor_state[2]**2)
            a = a[0]
            a = math.sqrt(a)
            b = math.sqrt((state[0]*1000)**2 + (state[2]*1000)**2)
            c = self.last_r * 1000

            if tools.is_triangle(a, b, c):
                angle = tools.triangle_angles(a, b, c)

                if angle < min_angle:
                    min_angle = angle
                    self.shot_point_in_pixel = armor2_in_pixel
                    self.shot_point_in_imu = armor_state

            else:
                min_angle = 0
                self.shot_point_in_pixel = armor2_in_pixel
                self.shot_point_in_imu = armor_state

        return self.shot_point_in_imu

    def getArmorPositionFromState(self, x):
        return self.h(x)[:3]

    def f(self, x, dt):
        # f - Process function
        x_new = x.copy()
        x_new[0] += x[4] * dt
        x_new[1] += x[5] * dt
        x_new[2] += x[6] * dt
        x_new[3] += x[7] * dt
        return x_new

    def j_f(self, x, dt):
        # J_f - Jacobian of process function
        dfdx = np.eye(9, 9)
        dfdx[0, 4] = dt
        dfdx[1, 5] = dt
        dfdx[2, 6] = dt
        dfdx[3, 7] = dt
        return dfdx

    def h(self, x):
        # h - Observation function
        z = np.zeros(4)
        xc, yc, zc, yaw, r = x[0], x[1], x[2], x[3], x[8]
        z[0] = xc - r * math.sin(yaw)  # xa
        z[1] = yc                      # ya
        z[2] = zc - r * math.cos(yaw)  # za
        z[3] = yaw                     # yaw
        return z

    def j_h(self, x):
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

    def handleArmorJump(self, a: Armor, max_match_distance):
        last_yaw = self.target_state[3]
        yaw = self.get_continous_yaw(a.yaw_in_imu_rad)

        if abs(yaw - last_yaw) > 0.4:
            print("Armor jump!")
            self.arrmor_jump = 1
            self.last_y = self.target_state[1]
            self.target_state[1] = np.reshape(a.in_imu_m, (3,))[1]
            self.target_state[3] = yaw
            self.target_state[8], self.last_r = self.last_r, self.target_state[8]

        current_p = np.reshape(a.in_imu_m, (3,))
        infer_p = self.getArmorPositionFromState(self.target_state)

        if np.linalg.norm(current_p - infer_p) > max_match_distance:
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
            print("y - error!!  :" + f"last_y: {self.last_y}" + "; " + f"y: {self.target_state[1]}")
    
    def updatePreState(self, preTime):
        self.pre_target_state = self.f(self.target_state, preTime)
        return self.pre_target_state


class BalanceInfantry(NormalRobot):
    '''平衡步兵(2装甲板)'''

    def __init__(self) -> None:
        super().__init__()

        self.target_type = "BalanceInfantry"

        # EKF
        self.q_v = [1e-2, 1e-2, 1e-2, 2e-2, 1, 5e-1, 7e-1, 4e-2, 1e-3]
        self.Q = np.diag(self.q_v)

        self.r_v = [1e-1, 1e-1, 1e-1, 6e-1]
        self.R = np.diag(self.r_v)

        self.P0 = np.eye(9)

        self.armors_in_pixel = deque(maxlen=2)

    def getPreShotPtsInImu(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0) -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        state = self.target_state

        flyTime = tools.getParaTime(state[:3] * 1000, bulletSpeed) / 1000

        state = self.f(state, deltatime+flyTime)  # predicted

        pre_armor_0 = np.array(self.getArmorPositionFromState(state)).reshape(3, 1) * 1000  # x y z

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] + math.pi
        _state[8] = self.last_r
        pre_armor_1 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        two_predict_points = [pre_armor_0, pre_armor_1]
        # print("aaaa{}".format(self.four_predict_points))

        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(yaw, pitch).T
        R_gimbal2camera = R_camera2gimbal.T

        # 得到2个可疑点的重投影点 armors_in_pixel
        # 与枪管的夹角
        min_angle = 180

        for armor_state in two_predict_points:

            # 调试用
            armor2_in_imu = armor_state
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3, 1)), np.zeros((3, 1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            self.armors_in_pixel.append(armor2_in_pixel)

            # 注意单位，单位为mm
            a = (armor_state[0]**2 + armor_state[2]**2)
            a = a[0]
            a = math.sqrt(a)
            b = math.sqrt((state[0]*1000)**2 + (state[2]*1000)**2)
            c = self.last_r * 1000

            if tools.is_triangle(a, b, c):
                angle = tools.triangle_angles(a, b, c)

                if angle < min_angle:
                    min_angle = angle
                    self.shot_point_in_pixel = armor2_in_pixel
                    self.shot_point_in_imu = armor_state

            else:
                min_angle = 0
                self.shot_point_in_pixel = armor2_in_pixel
                self.shot_point_in_imu = armor_state

        return self.shot_point_in_imu


class Outpost(NormalRobot):
    '''前哨站(3装甲板)'''

    def __init__(self) -> None:
        super().__init__()

        self.target_type = "Outpost"
        self.initial_r = 0.553/2  # (m)
        self.max_y_diff = 0.005

        self.target_state = np.zeros((8,))

        # EKF
        #            x      y     z    yaw  vx  vy    vz    vyaw
        self.q_v = [1e-2, 1e-2, 1e-2, 2e-2, 1, 5e-1, 7e-1, 4e-2]
        self.Q = np.diag(self.q_v)

        #            x     y     z    yaw
        self.r_v = [1e-1, 1e-1, 1e-1, 6e-1]
        self.R = np.diag(self.r_v)

        self.P0 = np.eye(8)

        self.armors_in_pixel = deque(maxlen=3)

    def init(self, a: Armor) -> None:
        self.armor = a
        self.armor_id = a.name
        xa, ya, za = np.reshape(a.in_imu_m, (3,))
        yaw = self.get_continous_yaw(a.yaw_in_imu_rad)

        # Set initial position at r behind the target
        r = self.initial_r
        xc = xa + r * math.sin(yaw)
        yc = ya
        zc = za + r * math.cos(yaw)

        self.target_state = np.zeros((8,))
        self.target_state[0] = xc
        self.target_state[1] = yc
        self.target_state[2] = zc
        self.target_state[3] = yaw

        self.last_y = yc
        self.last_yaw = 0

        self.ekfilter.setState(self.target_state)

        print(self.target_type + " --- Init EKF!")

    def j_f(self, x, dt):
        # J_f - Jacobian of process function
        dfdx = np.eye(8, 8)
        dfdx[0, 4] = dt
        dfdx[1, 5] = dt
        dfdx[2, 6] = dt
        dfdx[3, 7] = dt
        return dfdx

    def h(self, x):
        # h - Observation function
        z = np.zeros(4)
        xc, yc, zc, yaw, r = x[0], x[1], x[2], x[3], self.initial_r
        z[0] = xc - r * math.sin(yaw)  # xa
        z[1] = yc                      # ya
        z[2] = zc - r * math.cos(yaw)  # za
        z[3] = yaw                     # yaw
        return z

    def j_h(self, x):
        # J_h - Jacobian of observation function
        dhdx = np.zeros((4, 8))
        yaw, r = x[3], self.initial_r
        dhdx[0, 0] = dhdx[1, 1] = dhdx[2, 2] = dhdx[3, 3] = 1
        dhdx[0, 3] = -r * math.cos(yaw)
        dhdx[2, 3] = r * math.sin(yaw)
        return dhdx

    def getPreShotPtsInImu(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0) -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        state = self.target_state

        flyTime = tools.getParaTime(state[:3] * 1000, bulletSpeed) / 1000

        state = self.f(state, deltatime+flyTime)  # predicted

        pre_armor_0 = np.array(self.getArmorPositionFromState(state)).reshape(3, 1) * 1000  # x y z

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] + math.pi/3*2
        pre_armor_1 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        _state = state.copy()
        _state[1] = self.last_y
        _state[3] = state[3] - math.pi/3*2
        pre_armor_2 = np.array(self.getArmorPositionFromState(_state)).reshape(3, 1) * 1000

        three_predict_points = [pre_armor_0, pre_armor_1, pre_armor_2]
        # print("aaaa{}".format(self.four_predict_points))

        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(yaw, pitch).T
        R_gimbal2camera = R_camera2gimbal.T

        # 得到三个可疑点的重投影点 armors_in_pixel
        # 与枪管的夹角
        min_angle = 180

        for armor_state in three_predict_points:

            # 调试用
            armor2_in_imu = armor_state
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3, 1)), np.zeros((3, 1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            self.armors_in_pixel.append(armor2_in_pixel)

            # 注意单位，单位为mm
            a = (armor_state[0]**2 + armor_state[2]**2)
            a = a[0]
            a = math.sqrt(a)
            b = math.sqrt((state[0]*1000)**2 + (state[2]*1000)**2)
            c = self.initial_r * 1000

            if tools.is_triangle(a, b, c):
                angle = tools.triangle_angles(a, b, c)

                if angle < min_angle:
                    min_angle = angle
                    self.shot_point_in_pixel = armor2_in_pixel
                    self.shot_point_in_imu = armor_state

            else:
                min_angle = 0
                self.shot_point_in_pixel = armor2_in_pixel
                self.shot_point_in_imu = armor_state

        return self.shot_point_in_imu

    def handleArmorJump(self, a: Armor, max_match_distance):
        last_yaw = self.target_state[3]
        yaw = self.get_continous_yaw(a.yaw_in_imu_rad)

        if abs(yaw - last_yaw) > 0.4:
            print("Armor jump!")
            self.arrmor_jump = 1
            self.last_y = self.target_state[1]
            self.target_state[1] = np.reshape(a.in_imu_m, (3,))[1]
            self.target_state[3] = yaw

        current_p = np.reshape(a.in_imu_m, (3,))
        infer_p = self.getArmorPositionFromState(self.target_state)

        if np.linalg.norm(current_p - infer_p) > max_match_distance:
            print("State wrong!")
            self.state_error = 1
            r = self.initial_r
            self.target_state[0] = current_p[0] + r * math.sin(yaw)
            self.target_state[2] = current_p[2] + r * math.cos(yaw)
            self.target_state[4] = 0
            self.target_state[5] = 0
            self.target_state[6] = 0

        self.ekfilter.setState(self.target_state)

    def limitStateValue(self):
        # 前哨站的角速度有三种可能：0\0.2\0.4 rad/s，方向随机
        self.target_state[7] = tools.find_closest_value(self.target_state[7], (-0.4, -0.2, 0, 0.2, 0.4))

        if (self.last_y - self.target_state[1]) > self.max_y_diff:
            print("y - error!!")


class Base(NormalRobot):
    '''基地(单个静止装甲板)'''

    def __init__(self) -> None:
        super().__init__()

        self.target_type = "Base"

    def init(self, a: Armor) -> None:
        self.armor = a
        self.armor_id = a.name

        self.armors_in_pixel = deque(maxlen=1)

        print(self.target_type + " --- Init EKF!")

    def forwardPredict(self, dt):
        return None

    def setTargetState(self, state):
        pass

    def update(self, matched_armor: Armor) -> None:
        self.armor = matched_armor

    def getPreShotPtsInImu(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0) -> np.ndarray(shape=(3,)):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        state = self.armor.in_imu_m

        four_predict_points = [self.armor.in_imu_mm]
        # print("aaaa{}".format(self.four_predict_points))

        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(yaw, pitch).T
        R_gimbal2camera = R_camera2gimbal.T

        # 得到1个可疑点的重投影点 armors_in_pixel
        # 与枪管的夹角
        min_angle = 180

        for armor_state in four_predict_points:

            # 调试用
            armor2_in_imu = armor_state
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3, 1)), np.zeros((3, 1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            self.armors_in_pixel.append(armor2_in_pixel)

        return self.armor.in_imu_mm

    def getArmorPositionFromState(self, x):
        return self.armor.in_imu_m

    def get_continous_yaw(self, yaw):
        return 0

    def handleArmorJump(self, a: Armor, max_match_distance):
        self.armor = a

    def limitStateValue(self):
        pass
