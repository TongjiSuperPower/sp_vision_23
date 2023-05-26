import numpy as np
from numpy.typing import NDArray
from typing import TypeAlias, Callable


Matrix: TypeAlias = NDArray
ColumnVector: TypeAlias = NDArray


class ExtendedKalmanFilter:
    def __init__(
        self,
        f: Callable[[ColumnVector, float], ColumnVector],
        jacobian_f: Callable[[ColumnVector, float], Matrix],
        x0: ColumnVector,
        P0: Matrix,
        Q: Matrix,
        x_add: Callable[[ColumnVector, ColumnVector], ColumnVector] = None,
    ) -> None:
        '''
        f: 状态转移函数 f(x, dt_s) -> x
        jacobian_f: 状态转移函数的雅可比矩阵 jacobian_f(x, dt_s) -> F
        x0: 初始状态
        P0: 初始状态噪声
        Q: 过程噪声
        x_add: 定义状态向量加法, 便于处理角度突变
        '''
        self.f = f
        self.jacobian_f = jacobian_f
        self.x = x0
        self.P = P0
        self.Q = Q
        self._x_add = x_add

    def predict(self, dt_s: float) -> None:
        self.x = self.f(self.x, dt_s)
        F = self.jacobian_f(self.x, dt_s)
        self.P = F @ self.P @ F.T + self.Q

    def update(
        self,
        z: ColumnVector,
        h: Callable[[ColumnVector, tuple], ColumnVector],
        jacobian_h: Callable[[ColumnVector, tuple], Matrix],
        R: Matrix,
        z_subtract: Callable[[ColumnVector, ColumnVector, tuple], ColumnVector] = None
    ) -> None:
        '''
        z: 量测向量
        h: 量测函数 h(x) -> z
        jacobian_h: 量测函数的雅可比矩阵 jacobian_h(x) -> H
        R: 量测噪声
        z_subtract: 定义量测向量减法, 便于处理角度突变
        '''
        if z_subtract is None:
            z_subtract = np.subtract
        
        H = jacobian_h(self.x)
        K = self.P @ H.T @ np.linalg.inv(H @ self.P @ H.T + R)

        y = z_subtract(z, h(self.x))

        x_add = np.add if self._x_add is None else self._x_add
        self.x = x_add(self.x, K @ y)

        # Stable Compution of the Posterior Covariance
        # https://github.com/rlabbe/Kalman-and-Bayesian-Filters-in-Python/blob/master/07-Kalman-Filter-Math.ipynb
        I = np.identity(self.x.shape[0])
        self.P = (I - K @ H) @ self.P @ (I - K @ H).T + K @ R @ K.T
