import os
import sys
import numpy as np
from scipy.spatial.transform import Rotation as R
import math
import cv2
import toml

class EKF():
    '''EKF类,支持2维(xz)和3维(xyz)状态量'''
    def __init__(self, _stateDimension, _measurementDimension) -> None:
        self.stateDimension = _stateDimension
        self.measurementDimension = _measurementDimension

        self.state = np.zeros((self.stateDimension,1)) # 状态量，目标在世界坐标系下的位置和速度
        self.measurement = np.zeros((self.measurementDimension,1))

        self.rotationMatrix = None # 云台坐标系旋转矩阵C_b^n
        self.qMatrix, self.rrMatrix = self.readEKFConfig() # 过程噪声矩阵和观测噪声矩阵
        self.rMatrix = None # 转换后的观测噪声矩阵（由rr矩阵计算得到）
        self.pMatrix = None # 预测值的协方差矩阵

        # 创建观测矩阵
        self.hMatrix = np.zeros((self.measurementDimension, self.stateDimension))
        for i in range(self.measurementDimension):
            self.hMatrix[i, 2*i] = 1

        self.first = True

    def readEKFConfig(self):
        '''读取配置文件'''    
        cfgFile = 'assets/EKFConfig.toml'

        if not os.path.exists(cfgFile):
            print(cfgFile + ' not found')
            sys.exit(-1)
        
        content = toml.load(cfgFile) 

        Q = np.float32(content['Q'])  
        Rr = np.float32(content['Rr']) 

        return Q, Rr

    def step(self, deltaT, gesture, _state, observation, ptsInCam):
        '''EKF更新一个周期。deltaT:时间差值,gesture:云台的yaw和pitch值,_state:当前时刻的状态量,observation:观测量z、α、β'''          
        # 创建状态转移方程中的系数矩阵
        fMatrix = np.eye(self.stateDimension) # 状态转移矩阵
  
        gammaMatrix = np.zeros((self.stateDimension, self.measurementDimension)) # 过程噪声系数矩阵
        
        for i in range(int(self.stateDimension/2)):
            fMatrix[2*i, 2*i+1] = deltaT
            gammaMatrix[2*i, i] = deltaT*deltaT/2
            gammaMatrix[2*i+1, i] = deltaT       
        k = _state
        # pridict:
        # 更新x_k
        if self.first:
            self.state = _state.copy()
        else:
            self.state = fMatrix @ self.state.copy() 
        
        # correct:
        # 更新P_k
        if self.first:
            self.pMatrix = gammaMatrix @ self.qMatrix @ gammaMatrix.T
            self.first = False
        else:
            self.pMatrix = fMatrix @ self.pMatrix @ fMatrix.T + gammaMatrix @ self.qMatrix @ gammaMatrix.T
            
        # 计算R_k矩阵   
        [yaw,pitch] = gesture     
        yRotationMatrix = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
        xRotationMatrix = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
        self.rotationMatrix = yRotationMatrix @ xRotationMatrix

        [z, alpha, beta] = observation

        gMatrix = np.array([
            [math.tan(alpha), z/(math.cos(alpha)**2), 0], 
            [math.tan(beta), 0, z/(math.cos(beta)**2)], 
            [1, 0, 0]
            ])

        self.rMatrix = self.rotationMatrix @ gMatrix @ self.rrMatrix @ gMatrix.T @ self.rotationMatrix.T

        # 更新卡尔曼增益K_k
        kGain = (self.pMatrix @ self.hMatrix.T) @ np.linalg.inv(self.hMatrix @ self.pMatrix @ self.hMatrix.T + self.rMatrix)
        print(self.pMatrix.max())
        # 更新状态量
        self.state = self.state.copy() + kGain @ (self.hMatrix @ _state - self.hMatrix @ self.state.copy())

        # 更新p矩阵
        self.pMatrix = (np.eye(self.stateDimension) - kGain @ self.hMatrix) @ self.pMatrix
        res = self.hMatrix @ self.state
        return res
    
    def predictInWorld(self, time):
        '''返回时间time后世界坐标系下目标位置坐标'''
        # 根据匀速直线运动模型计算世界坐标系下预测坐标值
        predictedPosInWorld = []
        for i in range(self.stateDimension):
            predictedPosInWorld[i] = self.state[i] + time * self.state[i + 1] 
        
        return predictedPosInWorld    
    
    def predict(self, time, bulletSpeed):
        '''返回时间time后云台应该旋转的yaw和pitch值'''
        # 世界坐标系->云台坐标系
        predictedPosInWorld = self.predictInWorld(time)
        predictedPosInTripod = np.linalg.inv(self.rotationMatrix) @ predictedPosInWorld

        # 弹道下坠补偿
        distance = np.linalg.norm(predictedPosInTripod) # 云台坐标系下的距离
        flyTime = distance/bulletSpeed # 子弹飞行时间
        dropDistance = 0.5 * 9.7940 * flyTime**2 # 下坠距离
        predictedPosInTripod[1] += dropDistance 

        # 坐标值->yaw、pitch
        [x,y,z] = predictedPosInTripod
        yaw = cv2.fastAtan2(x, z)
        yaw = yaw if yaw<180 else yaw-360
        pitch = cv2.fastAtan2(-y, math.sqrt(x**2 + z**2))
        pitch = pitch if pitch<180 else pitch-360

        return yaw, pitch







        




        






     


      
