import numpy as np
from math import tan
from modules.ekf import ExtendedKalmanFilter, ColumnVector, Matrix
from modules.autoaim.armor import Armor
from modules.autoaim.targets.target import Target, get_trajectory_rad_and_s


max_match_m = 0.1

P0 = np.diag([
    1e-1,  # x_imu
    4e1,   # vx
    1e-1,  # y_imu
    1e-1,  # vy
    1e-1,  # z_imu
    4e1,  # vz
])

Q = np.diag([
    1e-4,  # x_imu
    1e-3,  # vx
    1e-4,  # y_imu
    1e-3,  # vy
    1e-4,  # z_imu
    1e-3,  # vz
])

R = np.diag([
    1e-4,  # x_camera
    1e-4,  # y_camera
    1e-2,  # z_camera
])


def f(x: ColumnVector, dt_s: float) -> ColumnVector:
    return jacobian_f(x, dt_s) @ x


def jacobian_f(x: ColumnVector, dt_s: float) -> Matrix:
    t = dt_s
    F = np.float64([
        [1, t, 0, 0, 0, 0],  # x_imu
        [0, 1, 0, 0, 0, 0],  # vx
        [0, 0, 1, t, 0, 0],  # y_imu
        [0, 0, 0, 1, 0, 0],  # vy
        [0, 0, 0, 0, 1, t],  # z_imu
        [0, 0, 0, 0, 0, 1],  # vz
    ])
    return F


def h(x: ColumnVector, armor: Armor) -> ColumnVector:
    R_imu2gimbal = armor._R_gimbal2imu.T
    R_gimbal2camera = armor._R_camera2gimbal.T
    t_camera2gimbal = armor._t_camera2gimbal / 1e3  # armor._t_camera2gimbal单位是mm
    x_in_imu, _, y_in_imu, _, z_in_imu, _ = x.T[0]

    armor_in_imu = np.float64([[x_in_imu, y_in_imu, z_in_imu]]).T
    armor_in_gimbal = R_imu2gimbal @ armor_in_imu
    armor_in_camera = R_gimbal2camera @ (armor_in_gimbal - t_camera2gimbal)

    return armor_in_camera


def jacobian_h(x: ColumnVector, R_imu2camera: Matrix) -> Matrix:
    H = np.float64([
        [1, 0, 0, 0, 0, 0],  # x_imu
        [0, 0, 1, 0, 0, 0],  # y_imu
        [0, 0, 0, 0, 1, 0],  # z_imu
    ])
    return R_imu2camera @ H


def get_x0(armor: Armor) -> ColumnVector:
    x, y, z = armor.in_imu_mm.T[0]
    return np.float64([[x, 0, y, 0, z, 0]]).T


class Simple(Target):
    def init(self, armor: Armor, img_time_s: float) -> None:
        x0 = get_x0(armor)
        self._ekf = ExtendedKalmanFilter(f, jacobian_f, x0, P0, Q)
        self._last_time_s = img_time_s

    def update(self, armor: Armor) -> bool:
        predicted_armor_in_camera = h(self._ekf.x, armor)
        error_m = np.linalg.norm(predicted_armor_in_camera - armor.in_camera_m)

        if error_m < max_match_m:
            self.init(armor, self._last_time_s)
            return True

        R_camera2imu = armor._R_gimbal2imu @ armor._R_camera2gimbal
        R_imu2camera = R_camera2imu.T
        self._ekf.update(armor.in_camera_m, lambda x: h(x, armor), lambda x: jacobian_h(x, R_imu2camera), R)
        return False

    def aim(self, bullet_speed_m_per_s: float) -> tuple[ColumnVector, float | None]:
        x_in_imu, _, y_in_imu, _, z_in_imu, _ = self._ekf.x.T[0]
        armor_in_imu = np.float64([[x_in_imu, y_in_imu, z_in_imu]]).T

        _, fly_time_s = get_trajectory_rad_and_s(armor_in_imu, bullet_speed_m_per_s)

        predicted_x = f(self._ekf.x, fly_time_s)
        x_in_imu, _, y_in_imu, _, z_in_imu, _ = predicted_x.T[0]
        aim_point_m = np.float64([[x_in_imu, y_in_imu, z_in_imu]]).T

        # 重力补偿
        gun_pitch_rad, _ = get_trajectory_rad_and_s(aim_point_m, bullet_speed_m_per_s)
        aim_point_m[1, 0] = -(aim_point_m[0, 0]**2 + aim_point_m[2, 0]**2)**0.5 * tan(gun_pitch_rad)

        return aim_point_m, 0.0

    def get_all_armor_positions_m(self) -> list[ColumnVector]:
        x_in_imu, _, y_in_imu, _, z_in_imu, _ = self._ekf.x.T[0]
        armor_in_imu = np.float64([[x_in_imu, y_in_imu, z_in_imu]]).T
        return [armor_in_imu]
