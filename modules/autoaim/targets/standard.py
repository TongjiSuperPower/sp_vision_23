import time
import numpy as np
from math import sin, cos, atan, pi, radians
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target, z_yaw_subtract, get_z_xyz, get_z_yaw, get_trajectory_rad_and_s, R_xyz, R_yaw


armor_num = 4

min_jump_rad = radians(60)
max_valid_rad = radians(10)
max_difference_m = 0.5
max_difference_rad = radians(20)

inital_r_m = 0.2
max_speed_rad_per_s = 10
P0 = np.diag([1e-2, 1e-2, 1e-2, 1e-2, 1, 1e-1, 1e-1, 4, 1, 4, max_speed_rad_per_s**2])
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-4, 1e-3, 1e-3, 1e-2, 1e-2, 1e-2, 1e-2])


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

    def update(self, armor: Armor) -> bool:
        z_xyz = get_z_xyz(armor)
        z_yaw = get_z_yaw(armor)

        self._last_z_yaw = z_yaw

        x = self._ekf.x.copy()
        old_yaw_rad = x[4, 0]

        use_r1_r2 = True
        yaw_valid = True
        error_rad = limit_rad(armor.yaw_in_imu_rad - old_yaw_rad)
        if error_rad > min_jump_rad:
            use_r1_r2 = False
            x[4, 0] = limit_rad(old_yaw_rad + 2*pi/armor_num)
        elif error_rad < -min_jump_rad:
            use_r1_r2 = False
            x[4, 0] = limit_rad(old_yaw_rad - 2*pi/armor_num)
        elif abs(error_rad) > max_valid_rad:
            yaw_valid = False

        self._ekf.x = x
        self._ekf.update(z_xyz, lambda x: h_xyz(x, use_r1_r2), lambda x: jacobian_h_xyz(x, use_r1_r2), R_xyz)

        if yaw_valid:
            self._ekf.update(z_yaw, h_yaw, jacobian_h_yaw, R_yaw, z_yaw_subtract)

        return False

    def aim(self, bullet_speed_m_per_s: float) -> tuple[ColumnVector, float]:
        pass

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        armor_positions_m: list[ColumnVector] = []
        x = self._ekf.x.copy()
        old_yaw_rad = x[4, 0]
        for i in range(armor_num):
            use_r1_r2 = (i%2 == 0)
            x[4, 0] = limit_rad(old_yaw_rad + i * 2 * pi / armor_num)
            z_xyz = h_xyz(x, use_r1_r2)
            armor_positions_m.append(z_xyz)
        return armor_positions_m
