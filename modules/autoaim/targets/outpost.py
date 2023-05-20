import math
import numpy as np
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.armor import Armor


radius_m = 0.2765
armor_num = 3

max_difference_m = 0.1
max_difference_rad = math.radians(30)
max_norm = (max_difference_m**2 + max_difference_rad**2)**0.5

inital_speed_rad_per_s = 0
max_speed_rad_per_s = 0.4 * 2 * math.pi
P0 = np.diag([1e-2, 1e-2, 1e-2, 4e-2, max_speed_rad_per_s**2])
Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
R = np.diag([8e-4, 5e-4, 8e-4, 4e-4])


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    F = np.float64([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, dt_s],
                    [0, 0, 0, 0, 1]])
    return F


def h(x: ColumnVector) -> ColumnVector:
    center_x_m, center_y_m, center_z_m, yaw_rad, _ = x.T[0]
    armor_x_m = center_x_m - radius_m * math.sin(yaw_rad)
    armor_y_m = center_y_m
    armor_z_m = center_z_m - radius_m * math.cos(yaw_rad)
    z = np.float64([[armor_x_m, armor_y_m, armor_z_m, yaw_rad]]).T
    return z


def jacobian_h(x: ColumnVector) -> Matrix:
    yaw_rad = x[3, 0]
    H = np.float64([[1, 0, 0, -radius_m * math.cos(yaw_rad), 0],
                    [0, 1, 0,                             0, 0],
                    [0, 0, 1,  radius_m * math.sin(yaw_rad), 0],
                    [0, 0, 0,                             1, 0]])
    return H


def x_add(x1: ColumnVector, x2: ColumnVector) -> ColumnVector:
    x3 = x1 + x2
    x3[3, 0] = limit_rad(x3[3, 0])
    return x3


def z_subtract(z1: ColumnVector, z2: ColumnVector) -> ColumnVector:
    z3 = z1 - z2
    z3[3, 0] = limit_rad(z3[3, 0])
    return z3


def h_inv(z: ColumnVector, speed_rad_per_s: float) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m, yaw_rad = z.T[0]
    center_x_m = armor_x_m + radius_m * math.sin(yaw_rad)
    center_y_m = armor_y_m
    center_z_m = armor_z_m + radius_m * math.cos(yaw_rad)
    x = np.float64([[center_x_m, center_y_m, center_z_m, yaw_rad, speed_rad_per_s]]).T
    return x


def get_z_from_armor(armor: Armor) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = armor.in_imu_m.T[0]
    z = np.float64([[armor_x_m, armor_y_m, armor_z_m, armor.yaw_in_imu_rad]]).T
    return z


def get_armor_position_m_from_x(x: ColumnVector) -> ColumnVector:
    z = h(x)
    armor_position_m = z[:3]
    return armor_position_m


class Outpost:
    def __init__(self) -> None:
        pass

    def init(self, armor: Armor) -> None:
        z = get_z_from_armor(armor)
        x0 = h_inv(z, inital_speed_rad_per_s)

        self._ekf = ExtendedKalmanFilter(
            f, h, jacobian_f, jacobian_h,
            x0, P0, Q, R,
            x_add, z_subtract
        )

    def predict(self, dt_s: float) -> None:
        self._ekf.predict(dt_s)

    def update(self, armor: Armor) -> None:
        z = get_z_from_armor(armor)
        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]

        min_norm = np.inf
        new_yaw_rad = old_yaw_rad
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * math.pi / armor_num)
            y = z_subtract(z, h(x))
            norm = np.linalg.norm(y)
            if norm < min_norm:
                min_norm = norm
                new_yaw_rad = x[3, 0]

        if min_norm < max_norm:
            x[3, 0] = new_yaw_rad
            self._ekf.x = x
            self._ekf.update(z)
        else:
            self.init(armor)

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        '''调试用'''
        armor_positions_m: list[ColumnVector] = []
        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * math.pi / armor_num)
            armor_position_m = get_armor_position_m_from_x(x)
            armor_positions_m.append(armor_position_m)
        return armor_positions_m
