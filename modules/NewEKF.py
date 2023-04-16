import numpy as np

class ExtendedKalmanFilter:
    def __init__(self, f, h, Jf, Jh, Q, R, P0):
        '''
        扩展卡尔曼滤波器类构造函数

        参数：
        f: 非线性状态转移函数
        h: 非线性观测函数
        Jf: 状态转移函数的雅可比矩阵
        Jh: 观测函数的雅可比矩阵
        Q: 状态转移噪声的协方差矩阵
        R: 观测噪声的协方差矩阵
        P0: 初始状态估计误差的协方差矩阵
        '''
        # 存储输入参数
        self.f = f
        self.h = h
        self.Jf = Jf
        self.Jh = Jh
        self.Q = Q
        self.R = R
        self.P_post = P0
        self.n = Q.shape[0]
        self.I = np.identity(self.n)
        self.x_pri = np.zeros(self.n)  # 预测的状态估计值
        self.x_post = np.zeros(self.n)  # 更新的状态估计值

    def set_state(self, x0):
        '''
        覆盖更新状态估计

        参数：
        x0: 状态值
        '''
        self.x_post = x0
    

    def predict(self,dt):
        '''
        使用非线性状态转移函数和其雅可比矩阵预测下一状态

        返回值：
        预测的状态估计值
        '''
        self.x_pri = self.f(self.x_post, dt)
        F = self.Jf(self.x_post, dt)
        self.P_pri = F @ self.P_post @ F.T + self.Q  # P_pri是预测的状态估计误差协方差矩阵

        # 存储预测的状态估计值和协方差矩阵以备更新使用
        self.x_post = self.x_pri
        self.P_post = self.P_pri

        return self.x_pri

    def update(self, z):
        '''
        使用非线性观测函数和其雅可比矩阵更新状态估计值

        参数：
        z: 观测值

        返回值：
        更新后的状态估计值
        '''
        H = self.Jh(self.x_pri)
        K = self.P_pri @ H.T @ np.linalg.inv(H @ self.P_pri @ H.T + self.R)  # 卡尔曼增益
        self.x_post = self.x_pri + K @ (z - self.h(self.x_pri))  # 更新后的状态估计值
        self.P_post = (self.I - K @ H) @ self.P_pri  # 更新后的状态估计误差协方差矩阵

        return self.x_post
