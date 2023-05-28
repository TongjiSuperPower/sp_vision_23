import time
import numpy as np
from math import sin, cos, tan, atan, pi, radians
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target, z_yaw_subtract, get_z_xyz, get_z_yaw, get_trajectory_rad_and_s, R_xyz, R_yaw


armor_num = 4

min_jump_rad = radians(60)
min_anittop_rad_per_s = 1.5  # 反小陀螺最小角速度
max_aim_yaw_rad = radians(20)  # 子弹射出方向与装甲板法线最大yaw

inital_r_m = 0.3
max_speed_rad_per_s = 10
P0 = np.diag([10, 1e-2, 1, 10, 1, 1e-1, 1e-1, 4, 1e-2, 4, max_speed_rad_per_s**2])
Q = np.diag([1e-2, 1e-4, 1e-4, 1e-2, 1e-2, 0, 0, 1, 1, 1, 1])


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    t = dt_s
    F = np.float64([
        [1, 0, 0, 0, 0, 0, 0, t, 0, 0, 0],  # x
        [0, 1, 0, 0, 0, 0, 0, 0, t, 0, 0],  # y1
        [0, 0, 1, 0, 0, 0, 0, 0, t, 0, 0],  # y2
        [0, 0, 0, 1, 0, 0, 0, 0, 0, t, 0],  # z
        [0, 0, 0, 0, 1, 0, 0, 0, 0, 0, t],  # yaw
        [0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0],  # r1
        [0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0],  # r2
        [0, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0],  # vx
        [0, 0, 0, 0, 0, 0, 0, 0, 1, 0, 0],  # vy
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 0],  # vz
        [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1]   # w
    ])
    return F


def h_xyz(x: ColumnVector, use_y1_r1: bool) -> ColumnVector:
    center_x_m, center_y1_m, center_y2_m, center_z_m, yaw_rad, r1_m, r2_m = x.T[0][:-4]
    center_y_m = center_y1_m if use_y1_r1 else center_y2_m
    r_m = r1_m if use_y1_r1 else r2_m
    armor_x_m = center_x_m - r_m * sin(yaw_rad)
    armor_y_m = center_y_m
    armor_z_m = center_z_m - r_m * cos(yaw_rad)
    z_xyz = np.float64([[armor_x_m, armor_y_m, armor_z_m]]).T
    return z_xyz


def jacobian_h_xyz(x: ColumnVector, use_y1_r1: bool) -> Matrix:
    yaw_rad, r1_m, r2_m = x.T[0][4:-4]
    r_m = r1_m if use_y1_r1 else r2_m
    if use_y1_r1:
        H = np.float64([[1, 0, 0, 0, -r_m*cos(yaw_rad), -sin(yaw_rad), 0, 0, 0, 0, 0],
                        [0, 1, 0, 0,                 0,             0, 0, 0, 0, 0, 0],
                        [0, 0, 0, 1,  r_m*sin(yaw_rad), -cos(yaw_rad), 0, 0, 0, 0, 0]])
    else:
        H = np.float64([[1, 0, 0, 0, -r_m*cos(yaw_rad), 0, -sin(yaw_rad), 0, 0, 0, 0],
                        [0, 0, 1, 0,                 0, 0,             0, 0, 0, 0, 0],
                        [0, 0, 0, 1,  r_m*sin(yaw_rad), 0, -cos(yaw_rad), 0, 0, 0, 0]])
    return H


def h_yaw(x: ColumnVector) -> ColumnVector:
    yaw_rad = x.T[0][4]
    z = np.float64([[yaw_rad]]).T
    return z


def jacobian_h_yaw(x: ColumnVector) -> Matrix:
    H = np.float64([[0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0]])
    return H


def x_add(x1: ColumnVector, x2: ColumnVector) -> ColumnVector:
    x3 = x1 + x2
    x3[4, 0] = limit_rad(x3[4, 0])
    return x3


def get_x0(z_xyz: ColumnVector, z_yaw: ColumnVector, r_m: float) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = z_xyz.T[0]
    yaw_rad = z_yaw[0, 0]
    center_x_m = armor_x_m + r_m * sin(yaw_rad)
    center_y_m = armor_y_m
    center_z_m = armor_z_m + r_m * cos(yaw_rad)
    x0 = np.float64([[center_x_m, center_y_m, center_y_m, center_z_m, yaw_rad, r_m, r_m, 0, 0, 0, 0]]).T
    return x0


class Standard(Target):
    def init(self, armor: Armor, img_time_s: float) -> None:
        z_xyz = get_z_xyz(armor)
        z_yaw = get_z_yaw(armor)
        x0 = get_x0(z_xyz, z_yaw, inital_r_m)

        self._ekf = ExtendedKalmanFilter(f, jacobian_f, x0, P0, Q, x_add)
        self._last_time_s = img_time_s
        self._last_z_yaw = z_yaw
        self._use_r1_r2 = True

    def update(self, armor: Armor) -> bool:
        z_xyz = get_z_xyz(armor)
        z_yaw = get_z_yaw(armor)

        self._last_z_yaw = z_yaw

        x = self._ekf.x.copy()
        old_yaw_rad = x[4, 0]

        error_rad = limit_rad(armor.yaw_in_imu_rad - old_yaw_rad)
        if error_rad > min_jump_rad:
            self._use_r1_r2 = not self._use_r1_r2
            x[4, 0] = limit_rad(old_yaw_rad + 2*pi/armor_num)
        elif error_rad < -min_jump_rad:
            self._use_r1_r2 = not self._use_r1_r2
            x[4, 0] = limit_rad(old_yaw_rad - 2*pi/armor_num)

        self._ekf.x = x
        self._ekf.update(z_xyz, lambda x: h_xyz(x, self._use_r1_r2), lambda x: jacobian_h_xyz(x, self._use_r1_r2), R_xyz)

        if abs(armor.yaw_in_camera_rad) < radians(15):
            self._ekf.update(z_yaw, h_yaw, jacobian_h_yaw, np.diag([8e-1]), z_yaw_subtract)
        else:
            self._ekf.update(z_yaw, h_yaw, jacobian_h_yaw, R_yaw, z_yaw_subtract)

        return False

    def aim(self, bullet_speed_m_per_s: float) -> tuple[ColumnVector, float | str | None]:
        current_time_s = time.time()
        current_state = f(self._ekf.x, current_time_s - self._last_time_s)
        current_yaw_rad = current_state[4, 0]

        # 近似估计子弹射出后目标的状态
        center_x, center_y1, center_y2, center_z = current_state[:4].T[0]
        center_m = np.float64([[center_x, min(center_y1, center_y2), center_z]]).T
        _, fly_to_center_s = get_trajectory_rad_and_s(center_m, bullet_speed_m_per_s)
        predicted_state = f(current_state, fly_to_center_s + 0.1)

        speed_rad_per_s = predicted_state[-1, 0]
        predicted_yaw_rad = predicted_state[4, 0]
        best_aim_yaw_rad = atan(predicted_state[0, 0] / predicted_state[3, 0])

        # 直接瞄准当前装甲板
        if abs(speed_rad_per_s) < min_anittop_rad_per_s:
            aim_point_m = h_xyz(predicted_state, self._use_r1_r2)

            if abs(limit_rad(predicted_yaw_rad - best_aim_yaw_rad)) < max_aim_yaw_rad:
                fire_time_s = 'now'
            else:
                fire_time_s = None

        # 反小陀螺
        else:
            if abs(limit_rad(predicted_yaw_rad - best_aim_yaw_rad)) > max_aim_yaw_rad:
                predicted_state[4, 0] = best_aim_yaw_rad
            aim_point_m = h_xyz(predicted_state, self._use_r1_r2)
            _, fly_to_armor_s = get_trajectory_rad_and_s(aim_point_m, bullet_speed_m_per_s)
            fly_to_armor_s += 0

            arrive_time_s = limit_rad(best_aim_yaw_rad - current_yaw_rad) / speed_rad_per_s

            fire_time_s = None
            if fly_to_armor_s < arrive_time_s:
                fire_time_s = arrive_time_s - fly_to_armor_s + current_time_s

        # 重力补偿
        gun_pitch_rad, _ = get_trajectory_rad_and_s(aim_point_m, bullet_speed_m_per_s)
        aim_point_m[1, 0] = -(aim_point_m[0, 0]**2 + aim_point_m[2, 0]**2)**0.5 * tan(gun_pitch_rad)

        return aim_point_m, fire_time_s

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        armor_positions_m: list[ColumnVector] = []
        x = self._ekf.x.copy()
        old_yaw_rad = x[4, 0]
        use_r1_r2 = self._use_r1_r2
        for i in range(armor_num):
            x[4, 0] = limit_rad(old_yaw_rad + i * 2 * pi / armor_num)
            z_xyz = h_xyz(x, use_r1_r2)
            armor_positions_m.append(z_xyz)
            use_r1_r2 = not use_r1_r2
        return armor_positions_m
