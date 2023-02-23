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
all_z = []
all_alpha = []
all_beta = []
for i in range(ptsInWorld.shape[0]):
    x = ptsInCam[i,0]
    y = ptsInCam[i,1]
    z = ptsInCam[i,2]
    alpha = math.atan(x/z)
    beta = math.atan(y/z)
    observation = [z, alpha, beta]

    all_z.append(z)
    all_alpha.append(alpha)
    all_beta.append(beta)
    #print(observation)

    deltaTime = 0.033

    if ekf.first==False:
        state[1,0] = (ptsInWorld[i,0] - ptsInWorld[i-1,0])/deltaTime
        state[3,0] = (ptsInWorld[i,1] - ptsInWorld[i-1,1])/deltaTime
        state[5,0] = (ptsInWorld[i,2] - ptsInWorld[i-1,2])/deltaTime
    
    state[0,0] = ptsInWorld[i,0]
    state[2,0] = ptsInWorld[i,1]
    state[4,0] = ptsInWorld[i,2]

    #print(state)    

    predictedPtsInWorld = ekf.step(deltaTime, [yaw,pitch], state, observation, np.reshape(ptsInCam[i], (3,1)))
    ptsEKF[i] = predictedPtsInWorld.T


data = ptsInWorld
data1 = ptsEKF

print('世界坐标系x、y、z的方差：')
print(np.var(ptsInWorld[:,0]))
print(np.var(ptsInWorld[:,1]))
print(np.var(ptsInWorld[:,2]))
print('z、alpha、beta的方差：')
print(np.var(all_z))
print(np.var(all_alpha))
print(np.var(all_beta))

length = data.shape[0]

x = np.linspace(0, length-1, length)

plt.plot(x, data[:,0], x, data[:,1], x, data[:,2], x, data1[:,0], x, data1[:,1], x, data1[:,2])
plt.legend(['x','y','z','x1','y1','z1'])
plt.savefig('./assets/ptsInWorld.jpg')


 
mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
x=data[:,0]
y=data[:,2]
z=data[:,1]*(-1)

x1=data1[:,0]
y1=data1[:,2]
z1=data1[:,1]*(-1)

ax.plot(x,y,z, label='ori')
ax.plot(x1,y1,z1, label='ekf')
ax.legend()
 
plt.savefig('./assets/test.jpg')
plt.show()


