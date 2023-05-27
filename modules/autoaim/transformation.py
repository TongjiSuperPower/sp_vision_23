import cv2
import math
import numpy as np
from scipy.spatial.transform import Rotation
from modules.tools import limit_rad


class LazyPNP:
    def __init__(self) -> None:
        self._cameraMatrix: np.ndarray = None
        self._distCoeffs: np.ndarray = None
        self._points_2d: np.ndarray = None
        self._points_3d: np.ndarray = None

        self._tvecs: np.ndarray = None
        self._rvecs: np.ndarray = None
        self._errors: np.ndarray = None

    def _solve_pnp(self) -> None:
        _, self._rvecs, self._tvecs, self._errors = cv2.solvePnPGeneric(
            self._points_3d, self._points_2d, self._cameraMatrix, self._distCoeffs,
            flags=cv2.SOLVEPNP_IPPE
        )

    def lazy_solve_pnp(self, points_3d: np.ndarray, points_2d: np.ndarray, cameraMatrix: np.ndarray, distCoeffs: np.ndarray) -> None:
        self._cameraMatrix = cameraMatrix
        self._distCoeffs = distCoeffs
        self._points_2d = points_2d
        self._points_3d = points_3d

    @property
    def rvecs(self) -> np.ndarray:
        if self._rvecs is None:
            self._solve_pnp()
        return self._rvecs

    @property
    def tvecs(self) -> np.ndarray:
        if self._tvecs is None:
            self._solve_pnp()
        return self._tvecs

    @property
    def rvec(self) -> np.ndarray:
        if self._rvecs is None:
            self._solve_pnp()
        return self._rvecs[0]

    @property
    def tvec(self) -> np.ndarray:
        if self._tvecs is None:
            self._solve_pnp()
        return self._tvecs[0]


class LazyTransformation(LazyPNP):
    def __init__(self) -> None:
        super().__init__()

        self._R_camera2gimbal: np.ndarray = None
        self._t_camera2gimbal: np.ndarray = None
        self._R_gimbal2imu: np.ndarray = None

        self._in_camera_mm: np.ndarray = None
        self._in_imu_mm: np.ndarray = None
        self._yaw_in_camera_rad: float = None
        self._yaw_in_imu_rad: float = None
        self._pitch_in_imu_rad: float = None

    def _get_yaw_pitch_in_imu_rad(self, rvec: np.ndarray) -> None:
        R_armor2camera, _ = cv2.Rodrigues(rvec)
        R_armor2gimbal = R_armor2camera
        R_armor2imu = self._R_gimbal2imu @ R_armor2gimbal
        return Rotation.from_matrix(R_armor2imu).as_euler('YXZ')[:2]

    def _transform(self) -> None:
        # 获得装甲板在imu坐标系下的朝向以及其中心点在相机坐标系下的坐标
        self._in_camera_mm = self.tvec  # points_3d是以装甲板中心点为原点, 所以tvec即为装甲板中心点在相机坐标系下的坐标
        self._yaw_in_imu_rad, self._pitch_in_imu_rad = self._get_yaw_pitch_in_imu_rad(self.rvec)

        # 获得装甲板中心点在云台坐标系下的坐标
        in_gimbal_mm = self._R_camera2gimbal @ self._in_camera_mm + self._t_camera2gimbal

        # 获得装甲板中心点在imu坐标系下的坐标
        self._in_imu_mm = self._R_gimbal2imu @ in_gimbal_mm

    def lazy_transform(self, R_camera2gimbal: np.ndarray, t_camera2gimbal: np.ndarray, R_gimbal2imu: np.ndarray) -> None:
        self._R_camera2gimbal = R_camera2gimbal
        self._t_camera2gimbal = t_camera2gimbal
        self._R_gimbal2imu = R_gimbal2imu

    @property
    def in_camera_mm(self) -> np.ndarray:
        if self._in_camera_mm is None:
            self._transform()
        return self._in_camera_mm

    @property
    def in_camera_m(self) -> np.ndarray:
        return self.in_camera_mm * 1e-3

    @property
    def in_imu_mm(self) -> np.ndarray:
        if self._in_imu_mm is None:
            self._transform()
        return self._in_imu_mm

    @property
    def in_imu_m(self) -> np.ndarray:
        return self.in_imu_mm * 1e-3

    @property
    def yaw_in_camera_rad(self) -> float:
        if self._yaw_in_camera_rad is None:
            R_armor2camera, _ = cv2.Rodrigues(self.rvec)
            self._yaw_in_camera_rad = Rotation.from_matrix(R_armor2camera).as_euler('YXZ')[0]
        return self._yaw_in_camera_rad

    @property
    def yaw_in_camera_degree(self) -> float:
        return math.degrees(self.yaw_in_camera_rad)

    @property
    def yaw_in_imu_rad(self) -> float:
        if self._yaw_in_imu_rad is None:
            self._transform()
        return self._yaw_in_imu_rad

    @property
    def yaw_in_imu_degree(self) -> float:
        return math.degrees(self.yaw_in_imu_rad)

    @property
    def pitch_in_imu_rad(self) -> float:
        if self._pitch_in_imu_rad is None:
            self._transform()
        return self._pitch_in_imu_rad

    @property
    def pitch_in_imu_degree(self) -> float:
        return math.degrees(self.pitch_in_imu_rad)
