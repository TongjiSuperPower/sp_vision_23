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

        self.state = np.zeros((self.stateDimension,1)) # 状态量，目标在世界坐标系下的位置(mm)和速度(mm/ms)
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
        self.stepNumber = 0

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

    def check_symmetric(self, a, rtol=1e-05, atol=1e-08):
        return np.allclose(a, a.T, rtol=rtol, atol=atol)

    def step(self, deltaT, gesture, _state, observation, ptsInCam):
        '''EKF更新一个周期。deltaT:时间差值,gesture:云台的yaw和pitch值,_state:当前时刻的状态量,observation:观测量z、α、β'''   
        #print('step\n')       
        self.stepNumber += 1
        if self.first:
            self.first = False
        # 创建状态转移方程中的系数矩阵
        fMatrix = np.eye(self.stateDimension) # 状态转移矩阵
  
        gammaMatrix = np.zeros((self.stateDimension, self.measurementDimension)) # 过程噪声系数矩阵
        
        for i in range(int(self.stateDimension/2)):
            fMatrix[2*i, 2*i+1] = deltaT
            gammaMatrix[2*i, i] = deltaT*deltaT/2
            gammaMatrix[2*i+1, i] = deltaT       
        k = _state

        # 计算R_k矩阵   
        [yaw,pitch] = gesture     
        yRotationMatrix = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
        xRotationMatrix = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
        self.rotationMatrix = yRotationMatrix @ xRotationMatrix
        #print('step update\n')
        #print(self.rotationMatrix)

        [z, alpha, beta] = observation

        gMatrix = np.array([
            [math.tan(alpha), z/(math.cos(alpha)**2), 0], 
            [math.tan(beta), 0, z/(math.cos(beta)**2)], 
            [1, 0, 0]
            ])

        self.rMatrix = self.rotationMatrix @ gMatrix @ self.rrMatrix @ gMatrix.T @ self.rotationMatrix.T

        # pridict:
        # 更新x_k
        if self.stepNumber <= 2:
            self.state = _state.copy()
        else:
            self.state = fMatrix @ self.state.copy() 
        
        # correct:
        # 更新P_k
        if self.stepNumber <= 2:
            self.pMatrix = gammaMatrix @ self.qMatrix @ gammaMatrix.T
            return self.hMatrix @ self.state
            
        else:
            self.pMatrix = fMatrix @ self.pMatrix @ fMatrix.T + gammaMatrix @ self.qMatrix @ gammaMatrix.T

        # 更新卡尔曼增益K_k
        kGain = (self.pMatrix @ self.hMatrix.T) @ np.linalg.inv(self.hMatrix @ self.pMatrix @ self.hMatrix.T + self.rMatrix)
        
        # 更新状态量
        self.state = self.state.copy() + kGain @ (self.hMatrix @ _state - self.hMatrix @ self.state.copy())

        # 更新p矩阵
        self.pMatrix = (np.eye(self.stateDimension) - kGain @ self.hMatrix) @ self.pMatrix
        res = self.hMatrix @ self.state
        
        if(self.check_symmetric(self.pMatrix) == False):
            print(np.matrix(self.pMatrix))
        return res

    

    def getPredictedPtsInWorld(self, time):
        '''返回时间time后世界坐标系下目标位置坐标'''
        # 根据匀速直线运动模型计算世界坐标系下预测坐标值
        predictedPosInWorld = []
        for i in range(self.measurementDimension-1):
            predictedPosInWorld.append(self.state[i*2] + time * self.state[i*2 + 1])
        predictedPosInWorld.append(self.state[4])
        return np.reshape(predictedPosInWorld,(3,))  
    
    def getFlyTime(self, bulletSpeed):
        '''迭代法求出子弹飞行时间(ms)'''
        posO = np.reshape(self.hMatrix @ self.state, (3,))
        pos = np.reshape(self.hMatrix @ self.state, (3,))
        cnt = 0
        maxCnt = 15
        tol = 1e-2
        t=0

        vMatrix = np.zeros((self.measurementDimension, self.stateDimension))
        for i in range(self.measurementDimension):
            vMatrix[i, 2*i+1] = 1

        while True:
            tn = self.getParaTime(pos, bulletSpeed)
            deltaPos = np.reshape(vMatrix @ self.state, (3,)) * tn
            pos = posO + deltaPos
            deltaTime = tn-t
            if deltaTime<tol or cnt > maxCnt:
                break
            t = tn
            cnt += 1
        
        return tn
            

    def getParaTime(self, pos, bulletSpeed):
        '''用抛物线求子弹到目标位置的时间'''
        pos = np.reshape(pos, (3,))
        x = pos[0]
        y = pos[1]
        z = pos[2]        
        
        dxz = math.sqrt(x*x+z*z)
        a = 0.5*9.7940/1000*dxz*dxz/(bulletSpeed*bulletSpeed)
        b = dxz
        c = a - y

        res1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
        res2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)

        beta1 = math.atan(res1)
        beta2 = math.atan(res2)

        t1 = dxz/(bulletSpeed*math.cos(beta1))
        t2 = dxz/(bulletSpeed*math.cos(beta2))
        
        #t = math.sqrt(x**2+y**2+z**2)/bulletSpeed

        t = t1 if t1<t2 else t2

        return t
    
    def getCompensatedPtsInWorld(self, pts, deltaTime, bulletSpeed):
        '''输入当前世界坐标(mm)，输出一段时间后目标的世界坐标(即枪管应该指向的世界坐标)(包括弹道下坠补偿);
        deltaTime:系统延迟时间(ms)
        bulletSpeed:弹速(m/s)'''
        flyTime = self.getParaTime(pts, bulletSpeed)
        # flyTime = 40
        prePts = self.getPredictedPtsInWorld(flyTime+deltaTime) # 匀速直线模型计算的坐标
        dropDistance = 0.5 * 9.7940/1000 * flyTime**2
        prePts[1] -= dropDistance # 因为y轴方向向下，所以是减法
        return prePts


    
    def predict(self, time, bulletSpeed):
        '''返回时间time后云台应该旋转的相对yaw和pitch值'''        
        distance = np.linalg.norm(self.hMatrix @ self.state) # 世界坐标系下的距离(mm)
        flyTime = distance/bulletSpeed # 子弹飞行时间(ms)
        dropDistance = 0.5 * 9.7940/1000 * flyTime**2 # 下坠距离(mm)

        # 世界坐标系->云台坐标系       
        predictedPosInWorld = self.getPredictedPtsInWorld(time+flyTime) 
        predictedPosInTripod = np.linalg.inv(self.rotationMatrix) @ predictedPosInWorld

        # 弹道下坠补偿        
        predictedPosInTripod[1] -= dropDistance 

        # 坐标值->yaw、pitch
        [x,y,z] = np.reshape(predictedPosInTripod,[3,])
        x = float(x)
        y = float(y)
        z = float(z) 
        yaw = cv2.fastAtan2(x, z)
        yaw = yaw if yaw<180 else yaw-360
        pitch = cv2.fastAtan2(-y, math.sqrt(x**2 + z**2))
        pitch = pitch if pitch<180 else pitch-360

        return yaw, pitch

