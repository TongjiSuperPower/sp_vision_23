import numpy as np
from scipy.spatial.transform import Rotation as R
import math

class EKF():
    '''EKF类,支持2维(xy)和3维(xyz)状态量'''
    def __init__(self, _stateDimension, _measurementDimension) -> None:
        self.stateDimension = _stateDimension
        self.measurementDimension = _measurementDimension

        self.state = np.zeros((self.stateDimension,1))
        self.measurement = np.zeros((self.measurementDimension,1))

        self.gkMatrix = None # 线性化矩阵G_k
        self.cbnMatrix = None # 云台坐标系姿态矩阵C_b^n
        self.qMatrix = None # 过程噪声矩阵（需要调参）
        self.rrMatrix = None # 观测噪声矩阵（需要调参,.toml文件中的Rr）
        self.rMatrix = None # 转换后的观测噪声矩阵（由rr矩阵计算得到）
        self.pMatrix = None # 预测值的协方差矩阵

        self.first = True

    def step(self, deltaT, quaternion, _state, observation):
        '''EKF更新一个周期。deltaT:时间差值,quaternion:陀螺仪传来的四元数,_state:当前时刻的状态量,observation:观测量z、α、β'''          
        # 创建状态转移方程中的系数矩阵
        fMatrix = np.eye(self.stateDimension) # 状态转移矩阵
  
        gammaMatrix = np.zeros((self.stateDimension, self.measurementDimension)) # 过程噪声系数矩阵
        
        for i in range(self.stateDimension/2):
            fMatrix[2*i, 2*i+1] = deltaT
            gammaMatrix[2*i, i] = deltaT*deltaT/2
            gammaMatrix[2*i+1, i] = deltaT
        
        # 创建观测矩阵
        hMatrix = np.zeros((self.measurementDimension, self.stateDimension))
        for i in range(self.measurementDimension):
            hMatrix[i, 2*i] = 1
        
        # pridict:
        self.state = fMatrix*self.state # 更新x_k
        
        # correct:
        # 更新P_k
        if self.first:
            self.pMatrix = gammaMatrix*self.qMatrix*gammaMatrix.T
        else:
            self.pMatrix = fMatrix*self.pMatrix*fMatrix.T + gammaMatrix*self.qMatrix*gammaMatrix.T
            self.first = False

        # 更新卡尔曼增益K_k
        # 计算R_k矩阵        
        rotationMatrix = R.from_quat(quaternion).as_matrix()

        [z, alpha, beta] = observation

        gMatrix = np.array([
            [math.tan(alpha), z/(math.cos(alpha)**2), 0], 
            [math.tan(beta), 0, z/(math.cos(beta)**2)], 
            [1, 0, 0]
            ])

        self.rMatrix = rotationMatrix * gMatrix * self.rrMatrix * gMatrix.T * rotationMatrix.T

        kGain = (self.pMatrix*hMatrix.T)/(hMatrix*self.pMatrix*hMatrix.T + self.rMatrix)

        # 更新状态量
        self.state += kGain * (_state - hMatrix * self.state)

        # 更新p矩阵
        self.pMatrix = (np.eye(self.stateDimension) - kGain * hMatrix) * self.pMatrix
    
    def predict(self, time):
        '''返回时间time后世界坐标系下目标的yaw和pitch值'''
        # 根据匀速直线运动模型计算世界坐标系下预测坐标值
        predictedPosInWorld = []
        for i in range(self.stateDimension):
            predictedPosInWorld[i] = self.state[i] + time * self.state[i + 1]
        
        # 转换为yaw和pitch
        




        






     


      
