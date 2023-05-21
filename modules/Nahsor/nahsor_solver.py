import cv2
import numpy as np

from modules import tools
from modules.Nahsor.nahsor_marker import NahsorMarker
from modules.autoaim.transformation import LazyTransformation


class NahsorSolver:
    def __init__(self, cameraMatrix: np.ndarray, distCoeffs: np.ndarray, R_camera2gimbal: np.ndarray,
                 t_camera2gimbal: np.ndarray) -> None:
        self._cameraMatrix: np.ndarray = cameraMatrix
        self._distCoeffs: np.ndarray = distCoeffs
        self._R_camera2gimbal = R_camera2gimbal
        self._t_camera2gimbal = t_camera2gimbal

    def solve(self, nahsor: NahsorMarker, predict_time: float, yaw_degree: float, pitch_degree: float):
        R_gimbal2imu = tools.R_gimbal2imu(yaw_degree, pitch_degree)

        def lazy_solve(nahsor: NahsorMarker, predict_time: float):
            points_2d = nahsor.get_2d_predict_corners(predict_time)
            points_2d.append(nahsor.r_center)
            # 3D坐标由能量机关尺寸图计算出
            # 靶心:[0, 193.5, 0]
            # points_3d = np.float32([[-186, 36-193.5, 0],
            #                         [-160, 353-193.5, 0],
            #                         [160, 353-193.5, 0],
            #                         [186, 36-193.5, 0],
            #                         [0, -501-193.5, 0]])
            points_3d = np.float32([[0, -330-193.5, 0],
                                    [-186, 36-193.5, 0],
                                    [0, 382-193.5, 0],
                                    [186, 36-193.5, 0],
                                    [0, -501-193.5, 0]])

            LazyTrans = LazyTransformation()
            LazyTrans.lazy_solve_pnp(points_3d, points_2d, self._cameraMatrix, self._distCoeffs)
            LazyTrans.lazy_transform(self._R_camera2gimbal, self._t_camera2gimbal, R_gimbal2imu)

            return LazyTrans

        return lazy_solve(nahsor, predict_time)
