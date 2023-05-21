import math
import numpy as np
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.armor import Armor


radius_m = 0.2765
armor_num = 3

max_difference_m = 0.2
max_difference_rad = math.radians(20)

inital_speed_rad_per_s = 0
max_speed_rad_per_s = 0.4 * 2 * math.pi
P0 = np.diag([1e-2, 1e-2, 1e-2, 1, max_speed_rad_per_s**2])
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-3, 1e-4])
R_xyz = np.diag([1e-2, 5e-3, 1e-2])
R_yaw = np.diag([5e-2])


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    F = np.float64([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, dt_s],
                    [0, 0, 0, 0, 1]])
    return F


def h_xyz(x: ColumnVector) -> ColumnVector:
    center_x_m, center_y_m, center_z_m, yaw_rad, _ = x.T[0]
    armor_x_m = center_x_m - radius_m * math.sin(yaw_rad)
    armor_y_m = center_y_m
    armor_z_m = center_z_m - radius_m * math.cos(yaw_rad)
    z = np.float64([[armor_x_m, armor_y_m, armor_z_m]]).T
    return z


def jacobian_h_xyz(x: ColumnVector) -> Matrix:
    yaw_rad = x[3, 0]
    H = np.float64([[1, 0, 0, -radius_m * math.cos(yaw_rad), 0],
                    [0, 1, 0,                             0, 0],
                    [0, 0, 1,  radius_m * math.sin(yaw_rad), 0],])
    return H


def h_yaw(x: ColumnVector) -> ColumnVector:
    _, _, _, yaw_rad, _ = x.T[0]
    z = np.float64([[yaw_rad]]).T
    return z


def jacobian_h_yaw(x: ColumnVector) -> Matrix:
    H = np.float64([[0, 0, 0, 1, 0]])
    return H


def x_add(x1: ColumnVector, x2: ColumnVector) -> ColumnVector:
    x3 = x1 + x2
    x3[3, 0] = limit_rad(x3[3, 0])
    return x3


def z_yaw_subtract(z1: ColumnVector, z2: ColumnVector) -> ColumnVector:
    z3 = z1 - z2
    z3[0, 0] = limit_rad(z3[0, 0])
    return z3


def h_inv(z_xyz: ColumnVector, z_yaw: ColumnVector, speed_rad_per_s: float) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = z_xyz.T[0]
    yaw_rad = z_yaw[0, 0]
    center_x_m = armor_x_m + radius_m * math.sin(yaw_rad)
    center_y_m = armor_y_m
    center_z_m = armor_z_m + radius_m * math.cos(yaw_rad)
    x = np.float64([[center_x_m, center_y_m, center_z_m, yaw_rad, speed_rad_per_s]]).T
    return x


def get_z_xyz_from_armor(armor: Armor) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = armor.in_imu_m.T[0]
    z = np.float64([[armor_x_m, armor_y_m, armor_z_m]]).T
    return z


def get_z_yaw_from_armor(armor: Armor) -> ColumnVector:
    z = np.float64([[armor.yaw_in_imu_rad]]).T
    return z


class Outpost:
    def __init__(self) -> None:
        pass

    def init(self, armor: Armor) -> None:
        self.debug_yaw_rad = armor.yaw_in_imu_rad
        z_xyz = get_z_xyz_from_armor(armor)
        z_yaw = get_z_yaw_from_armor(armor)
        x0 = h_inv(z_xyz, z_yaw, inital_speed_rad_per_s)

        self._ekf = ExtendedKalmanFilter(f, jacobian_f, x0, P0, x_add)

    def predict(self, dt_s: float) -> None:
        self._ekf.predict(dt_s, Q)

    def update(self, armor: Armor) -> None:
        self.debug_yaw_rad = armor.yaw_in_imu_rad

        z_xyz = get_z_xyz_from_armor(armor)
        z_yaw = get_z_yaw_from_armor(armor)

        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]

        min_distance_m = np.inf
        new_yaw_rad = old_yaw_rad
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * math.pi / armor_num)
            distance_m = np.linalg.norm(z_xyz - h_xyz(x))
            if distance_m < min_distance_m:
                min_distance_m = distance_m
                new_yaw_rad = x[3, 0]

        if min_distance_m < max_difference_m:
            x[3, 0] = new_yaw_rad
            self._ekf.x = x
            self._ekf.update(z_xyz, h_xyz, jacobian_h_xyz, R_xyz)

            difference_rad = abs(limit_rad(new_yaw_rad - z_yaw[0, 0]))
            if difference_rad < max_difference_rad:
                self._ekf.update(z_yaw, h_yaw, jacobian_h_yaw, R_yaw, z_yaw_subtract)

        else:
            print('reinit!')
            self.init(armor)

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        '''调试用'''
        armor_positions_m: list[ColumnVector] = []
        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * math.pi / armor_num)
            armor_position_m = h_xyz(x)
            armor_positions_m.append(armor_position_m)
        return armor_positions_m
