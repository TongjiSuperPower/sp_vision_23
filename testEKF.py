# 测试基于扩展卡尔曼滤波（EKF）的装甲板运动预测功能
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
from modules.ExtendKF import EKF

filePath = './assets/ptsInCam.txt'

yaw = 12.0/180*math.pi
pitch = 2.0/180*math.pi
gesture = [yaw,pitch] # 弧度制

ptsInCam = np.loadtxt(filePath)

ptsInTripod = ptsInCam + np.array([0, 20, 10])

yRotationMatrix = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
xRotationMatrix = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])

# 云台坐标系->世界坐标系
ptsInWorld = np.zeros((ptsInTripod.shape))
i=0
for single in ptsInTripod:
    single = np.reshape(single, (3,1))    
    ptsInWorld[i] = np.dot(yRotationMatrix, np.dot(xRotationMatrix,single)).T
    i+=1

ptsEKF = np.zeros((ptsInWorld.shape))
ekf = EKF(6, 3)
state = np.zeros((6,1))
for i in range(ptsInWorld.shape[0]):
    x = ptsInCam[i,0]
    y = ptsInCam[i,1]
    z = ptsInCam[i,2]
    alpha = math.atan(x/z)
    beta = math.atan(y/z)
    observation = [z, alpha, beta]
    #print(observation)

    deltaTime = 0.03

    # ?状态量里的速度怎么计算

    if ekf.first==False:
        state[1,0] = (ptsInWorld[i,0] - state[0,0])/deltaTime
        state[3,0] = (ptsInWorld[i,1] - state[2,0])/deltaTime
        state[5,0] = (ptsInWorld[i,2] - state[4,0])/deltaTime
    
    state[0,0] = ptsInWorld[i,0]
    state[2,0] = ptsInWorld[i,1]
    state[4,0] = ptsInWorld[i,2]

    #print(state)    

    predictedPtsInWorld = ekf.step(deltaTime, [yaw,pitch], state, observation, np.reshape(ptsInCam[i], (3,1)))
    ptsEKF[i] = predictedPtsInWorld.T


data = ptsInWorld
data1 = ptsEKF


length = data.shape[0]

x = np.linspace(0, length-1, length)

plt.plot(x, data[:,0], x, data[:,1], x, data[:,2], x, data1[:,0], x, data1[:,1], x, data1[:,2])
plt.legend(['x','y','z','x1','y1','z1'])
plt.savefig('./assets/ptsInCam.jpg')
#plt.show()


 
mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
x=data[:,0]
y=data[:,2]
z=data[:,1]*(-1)

ax.plot(x[:100], y[:100], z[:100], label='parametric curve')
ax.legend()
 
plt.savefig('./assets/test.jpg')
#plt.show()


