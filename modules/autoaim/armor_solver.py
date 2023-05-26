import math
import numpy as np
from collections.abc import Iterable

import modules.tools as tools
from modules.autoaim.armor import Armor


lightbar_length, small_width, big_width = 56, 135, 230  # 真装甲板 单位mm
# lightbar_length, small_width, big_width = 70, 140, 230  # 假装甲板 单位mm


class ArmorSolver:
    def __init__(self, cameraMatrix: np.ndarray, distCoeffs: np.ndarray, R_camera2gimbal: np.ndarray, t_camera2gimbal: np.ndarray) -> None:
        self._cameraMatrix: np.ndarray = cameraMatrix
        self._distCoeffs: np.ndarray = distCoeffs
        self._R_camera2gimbal = R_camera2gimbal
        self._t_camera2gimbal = t_camera2gimbal

    def solve(self, armors: Iterable[Armor], yaw_degree: float, pitch_degree: float) -> Iterable[Armor]:
        R_gimbal2imu = tools.R_gimbal2imu(yaw_degree, pitch_degree)

        def lazy_solve(armor: Armor) -> Armor:
            width = big_width if 'big' in armor.name else small_width

            points_2d = armor.points
            points_3d = np.float32([[-width / 2, -lightbar_length / 2, 0],
                                    [width / 2, -lightbar_length / 2, 0],
                                    [width / 2, lightbar_length / 2, 0],
                                    [-width / 2, lightbar_length / 2, 0]])

            armor.lazy_solve_pnp(points_3d, points_2d, self._cameraMatrix, self._distCoeffs)
            armor.lazy_transform(self._R_camera2gimbal, self._t_camera2gimbal, R_gimbal2imu)

            return armor

        return (lazy_solve(armor) for armor in armors)
