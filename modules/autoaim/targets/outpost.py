import time
import numpy as np
from math import sin, cos, tan, atan, pi, radians
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.tools import limit_rad
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target, z_yaw_subtract, get_z_xyz, get_z_yaw, get_trajectory_rad_and_s, R_xyz, R_yaw


radius_m = 0.2765
armor_num = 3

max_difference_m = 0.5
max_difference_rad = radians(20)

max_speed_rad_per_s = 0.4 * 2 * pi
P0 = np.diag([1e-2, 1e-2, 1e-2, 1, max_speed_rad_per_s**2])
Q = np.diag([1e-4, 1e-4, 1e-4, 1e-4, 1e-3])


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    t = dt_s
    F = np.float64([[1, 0, 0, 0, 0],
                    [0, 1, 0, 0, 0],
                    [0, 0, 1, 0, 0],
                    [0, 0, 0, 1, t],
                    [0, 0, 0, 0, 1]])
    return F


def h_xyz(x: ColumnVector) -> ColumnVector:
    center_x_m, center_y_m, center_z_m, yaw_rad, _ = x.T[0]
    armor_x_m = center_x_m - radius_m * sin(yaw_rad)
    armor_y_m = center_y_m
    armor_z_m = center_z_m - radius_m * cos(yaw_rad)
    z_xyz = np.float64([[armor_x_m, armor_y_m, armor_z_m]]).T
    return z_xyz


def jacobian_h_xyz(x: ColumnVector) -> Matrix:
    yaw_rad = x[3, 0]
    H = np.float64([[1, 0, 0, -radius_m * cos(yaw_rad), 0],
                    [0, 1, 0,                        0, 0],
                    [0, 0, 1,  radius_m * sin(yaw_rad), 0],])
    return H


def h_yaw(x: ColumnVector) -> ColumnVector:
    _, _, _, yaw_rad, _ = x.T[0]
    z_yaw = np.float64([[yaw_rad]]).T
    return z_yaw


def jacobian_h_yaw(x: ColumnVector) -> Matrix:
    H = np.float64([[0, 0, 0, 1, 0]])
    return H


def x_add(x1: ColumnVector, x2: ColumnVector) -> ColumnVector:
    x3 = x1 + x2
    x3[3, 0] = limit_rad(x3[3, 0])
    return x3


def get_x0(z_xyz: ColumnVector, z_yaw: ColumnVector) -> ColumnVector:
    armor_x_m, armor_y_m, armor_z_m = z_xyz.T[0]
    yaw_rad = z_yaw[0, 0]
    center_x_m = armor_x_m + radius_m * sin(yaw_rad)
    center_y_m = armor_y_m
    center_z_m = armor_z_m + radius_m * cos(yaw_rad)
    x0 = np.float64([[center_x_m, center_y_m, center_z_m, yaw_rad, 0]]).T
    return x0


class Outpost(Target):
    def init(self, armor: Armor, img_time_s: float) -> None:
        z_xyz = get_z_xyz(armor)
        z_yaw = get_z_yaw(armor)
        x0 = get_x0(z_xyz, z_yaw)

        self._ekf = ExtendedKalmanFilter(f, jacobian_f, x0, P0, Q, x_add)
        self._last_time_s = img_time_s
        self._last_z_yaw = z_yaw

    def update(self, armor: Armor) -> bool:
        z_xyz = get_z_xyz(armor)
        z_yaw = get_z_yaw(armor)

        self._last_z_yaw = z_yaw

        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]

        min_norm = np.inf
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * pi / armor_num)
            distance_m = np.linalg.norm(z_xyz - h_xyz(x))
            norm = (distance_m**2 + limit_rad(x[3, 0] - z_yaw[0, 0])**2)**0.5
            if norm < min_norm:
                min_norm = norm
                min_distance_m = distance_m
                new_yaw_rad = x[3, 0]

        if min_distance_m < max_difference_m:
            reinit = False

            x[3, 0] = new_yaw_rad
            self._ekf.x = x
            self._ekf.update(z_xyz, h_xyz, jacobian_h_xyz, R_xyz)

            difference_rad = abs(limit_rad(new_yaw_rad - z_yaw[0, 0]))
            if difference_rad < max_difference_rad:
                self._ekf.update(z_yaw, h_yaw, jacobian_h_yaw, R_yaw, z_yaw_subtract)

        else:
            reinit = True
            self.init(armor, self._last_time_s)

        return reinit

    def aim(self, bullet_speed_m_per_s: float) -> tuple[ColumnVector, float]:
        speed_rad_per_s = self._ekf.x[4, 0]

        if abs(speed_rad_per_s) < max_speed_rad_per_s / 100:
            aim_point_m = h_xyz(self._ekf.x)
            fire_time_s = None

        else:
            center_in_imu_m = self._ekf.x[:3]
            center_x, _, center_z = center_in_imu_m.T[0]
            aim_yaw_rad = atan(center_x / center_z)

            current_time_s = time.time()
            x = f(self._ekf.x, current_time_s - self._last_time_s)
            current_yaw_rad = x[3, 0]

            x[3, 0] = aim_yaw_rad
            aim_point_m = h_xyz(x)
            _, fly_time_s = get_trajectory_rad_and_s(aim_point_m, bullet_speed_m_per_s)
            fly_time_s += 1.05

            arrive_time_s = limit_rad(aim_yaw_rad - current_yaw_rad) / speed_rad_per_s
            rotate_to_next_time_s = 2 * pi / armor_num / abs(speed_rad_per_s)

            fire_time_s = None
            for _ in range(3):
                arrive_time_s += rotate_to_next_time_s
                if fly_time_s < arrive_time_s:
                    fire_time_s = arrive_time_s - fly_time_s + current_time_s
                    break

        # 重力补偿
        gun_pitch_rad, _ = get_trajectory_rad_and_s(aim_point_m, bullet_speed_m_per_s)
        aim_point_m[1, 0] = -(aim_point_m[0, 0]**2 + aim_point_m[2, 0]**2)**0.5 * tan(gun_pitch_rad)

        return aim_point_m, fire_time_s

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        armor_positions_m: list[ColumnVector] = []
        x = self._ekf.x.copy()
        old_yaw_rad = x[3, 0]
        for i in range(armor_num):
            x[3, 0] = limit_rad(old_yaw_rad + i * 2 * pi / armor_num)
            z_xyz = h_xyz(x)
            armor_positions_m.append(z_xyz)
        return armor_positions_m
