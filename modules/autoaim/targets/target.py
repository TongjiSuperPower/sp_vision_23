import numpy as np
from math import sin, cos, atan, sqrt
from modules.ekf import ExtendedKalmanFilter, ColumnVector
from modules.autoaim.armor import Armor
from modules.tools import limit_rad


R_xyz = np.diag([8e-2, 8e-2, 8e-2])
R_yaw = np.diag([1.0])


def z_yaw_subtract(z1: ColumnVector, z2: ColumnVector) -> ColumnVector:
    z3 = z1 - z2
    z3[0, 0] = limit_rad(z3[0, 0])
    return z3


def get_z_xyz(armor: Armor) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = armor.in_imu_m.T[0]
    z_xyz = np.float64([[armor_x_m, armor_y_m, armor_z_m]]).T
    return z_xyz


def get_z_yaw(armor: Armor) -> ColumnVector:
    z_yaw = np.float64([[armor.yaw_in_imu_rad]]).T
    return z_yaw


def get_trajectory_rad_and_s(position_m: ColumnVector, bullet_speed_m_per_s: float) -> tuple[float, float]:
    x, y, z = position_m.T[0]

    g = 9.794
    distance_m = (x**2 + z**2)**0.5

    a = 0.5 * g * distance_m**2 / bullet_speed_m_per_s**2
    b = -distance_m
    c = a - y
    result1 = (-b + sqrt(b**2-4*a*c))/(2*a)
    result2 = (-b - sqrt(b**2-4*a*c))/(2*a)

    pitch1 = atan(result1)
    pitch2 = atan(result2)
    t1 = distance_m / (bullet_speed_m_per_s * cos(pitch1))
    t2 = distance_m / (bullet_speed_m_per_s * cos(pitch2))

    pitch_rad = pitch1 if t1 < t2 else pitch2
    fly_time_s = t1 if t1 < t2 else t2

    return pitch_rad, fly_time_s


class Target:
    def __init__(self) -> None:
        self._last_time_s: float = None
        self._ekf: ExtendedKalmanFilter = None

    def init(self, armor: Armor, img_time_s: float) -> None:
        raise NotImplementedError('该函数需子类实现')

    def predict(self, img_time_s: float) -> None:
        dt_s = img_time_s - self._last_time_s
        self._last_time_s = img_time_s
        self._ekf.predict(dt_s)

    def update(self, armor: Armor) -> bool:
        raise NotImplementedError('该函数需子类实现')

    def aim(self, bullet_speed_m_per_s: float) -> tuple[ColumnVector, float | None]:
        raise NotImplementedError('该函数需子类实现')

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        raise NotImplementedError('该函数需子类实现')
