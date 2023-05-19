import math
import numpy as np
from numpy.typing import NDArray
from collections.abc import Iterable
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.targets.target import Target
from modules.autoaim.armor import Armor


radius_m = 0.2765
angle_between_armor_rad = 2 * math.pi / 3
inital_speed_rad_per_s = 0.2 * 2 * math.pi

P0 = np.diag([1e-2, 1e-2, 1e-2, 4e-2, 2])
Q = np.diag([1e-6, 1e-6, 1e-6, 1e-6, 1e-6])
R = np.diag([8e-4, 5e-4, 8e-4, 4e-4])

def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    F = np.float64([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, dt_s],
                    [0, 0, 0, 0, 1]])
    return F


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


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
    armor_in_imu_m = z[:3]
    return armor_in_imu_m


class Outpost(Target):
    def __init__(self, armor: Armor, time_s: float) -> None:
        self.name = armor.name
        self._last_time_s = time_s

        z = get_z_from_armor(armor)
        x0 = h_inv(z, inital_speed_rad_per_s)

        self._ekf = ExtendedKalmanFilter(
            f, h, jacobian_f, jacobian_h,
            x0, P0, Q, R,
            x_add, z_subtract
        )

    def predict_to(self, time_s: float) -> None:
        dt_s = time_s - self._last_time_s
        self._last_time_s = time_s
        self._ekf.predict(dt_s)

    def get_armor_position_m(self) -> ColumnVector:
        return get_armor_position_m_from_x(self._ekf.x)

    def get_all_armor_positions_m(self) -> Iterable[ColumnVector]:
        def armor_position_generator() -> Iterable[ColumnVector]:
            x = self._ekf.x.copy()
            for _ in range(3):
                x[3, 0] = limit_rad(x[3, 0] + angle_between_armor_rad)
                armor_position_m = get_armor_position_m_from_x(x)
                yield armor_position_m
        return armor_position_generator()

    def update(self, armor: Armor) -> None:
        z = get_z_from_armor(armor)
        self._ekf.update(z)

    def handle_armor_jump(self, armor: Armor) -> None:
        x = self._ekf.x.copy()

        min_distance_m = np.inf
        new_yaw_rad: ColumnVector = None
        for _ in range(3):
            x[3, 0] = limit_rad(x[3, 0] + angle_between_armor_rad)

            armor_in_imu_m = get_armor_position_m_from_x(x)
            distance_m = np.linalg.norm(armor_in_imu_m - armor.in_imu_m)

            if distance_m < min_distance_m:
                min_distance_m = distance_m
                new_yaw_rad = x[3, 0]

        self._ekf.x[3, 0] = new_yaw_rad
        self.update(armor)
