import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias, Callable


Matrix: TypeAlias = NDArray
ColumnVector: TypeAlias = NDArray


class ExtendedKalmanFilter:
    def __init__(
        self,
        f: Callable[[ColumnVector, float], ColumnVector],
        h: Callable[[ColumnVector], ColumnVector],
        jacobian_f: Callable[[ColumnVector, float], Matrix],
        jacobian_h: Callable[[ColumnVector], Matrix],
        x0: ColumnVector,
        P0: Matrix,
        Q: Matrix,
        R: Matrix,
        x_add: Callable[[ColumnVector, ColumnVector], ColumnVector] = None,
        z_subtract: Callable[[ColumnVector, ColumnVector], ColumnVector] = None
    ) -> None:
        '''
        f: 状态转移函数 f(x, dt_s) -> x
        h: 量测函数 h(x) -> z
        jacobian_f: 状态转移函数的雅可比矩阵 jacobian_f(x, dt_s) -> F
        jacobian_h: 量测函数的雅可比矩阵 jacobian_h(x) -> H
        x0: 初始状态
        P0: 初始状态噪声
        Q: 过程噪声
        R: 量测噪声
        x_add: 定义状态向量加法, 便于处理角度突变
        z_subtract: 定义量测向量减法, 便于处理角度突变
        '''
        self.f = f
        self.h = h
        self.jacobian_f = jacobian_f
        self.jacobian_h = jacobian_h
        self.x = x0
        self.P = P0
        self.Q = Q
        self.R = R
        self._x_add = x_add
        self._z_subtract = z_subtract

    def predict(self, dt_s: float) -> None:
        self.x = self.f(self.x, dt_s)
        F = self.jacobian_f(self.x, dt_s)
        self.P = F @ self.P @ F.T + self.Q

    def update(self, z: ColumnVector) -> None:
        H = self.jacobian_h(self.x)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + self.R)

        z_subtract = np.subtract if self._z_subtract is None else self._z_subtract
        y = z_subtract(z, self.h(self.x))

        x_add = np.add if self._x_add is None else self._x_add
        self.x = x_add(self.x, K @ y)

        # Stable Compution of the Posterior Covariance
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
        I = np.identity(self.x.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ self.R @ K.T
