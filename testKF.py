# 测试基于filterpy中线性kf的装甲板运动预测功能
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
from modules.ExtendKF import EKF
import filterpy
from filterpy.kalman import KalmanFilter
from scipy.linalg import block_diag
from filterpy.common import Q_discrete_white_noise
deltaTime = 5 # 单位是ms
filePath = './assets/ptsInCam.txt'

yaw = 0/180*math.pi
pitch = 0/180*math.pi
gesture = [yaw,pitch] # 弧度制

ptsInCam = np.loadtxt(filePath) # mm

ptsInTripod = ptsInCam + np.array([0, 60, 50])

yRotationMatrix = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
xRotationMatrix = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])

# 云台坐标系->世界坐标系
ptsInWorld = np.zeros((ptsInTripod.shape))
i=0
for single in ptsInTripod:
    single = np.reshape(single, (3,1))    
    ptsInWorld[i] = np.dot(yRotationMatrix, np.dot(xRotationMatrix,single)).T
    i+=1

# KF:
R_std = 0.35
Q_std = 0.04

def tracker1():
    
    tracker = KalmanFilter(dim_x=6,dim_z=3)
    tracker.F = np.array([[1,deltaTime,0,0,0,0],
                        [0,1,0,0,0,0],
                        [0,0,1,deltaTime,0,0],
                        [0,0,0,1,0,0],
                        [0,0,0,0,1,deltaTime],
                        [0,0,0,0,0,1]])
    tracker.u=0.
    # Q矩阵-过程噪声矩阵
    q = Q_discrete_white_noise(dim=3, dt=deltaTime, var=Q_std**2)
    tracker.Q = block_diag(q, q)
    print(tracker.Q)
    # H矩阵-量测矩阵
    tracker.H = np.array([[1, 0, 0,        0,0,0],
                        [0,        0, 1, 0,0,0],
                        [0,0,0,0,1,0]])
    # R矩阵-量测噪声矩阵
    tracker.R = np.eye(3) * R_std**2
    
    tracker.x = np.array([[ptsInWorld[1][0], 
                           (ptsInWorld[2][0]-ptsInWorld[1][0])/deltaTime,
                             ptsInWorld[0][1],
                               (ptsInWorld[2][1]-ptsInWorld[1][1])/deltaTime,
                               ptsInWorld[0][2],
                               (ptsInWorld[2][2]-ptsInWorld[1][2])/deltaTime
                               ]]).T
    tracker.P = np.eye(6) * 500.
    return tracker

# run filter
robot_tracker = tracker1()
mu, cov, _, _ = robot_tracker.batch_filter(ptsInWorld[1:,:])

data = ptsInWorld[1:,:]
data1 = mu


length = data.shape[0]

x = np.linspace(0, length-1, length)
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.plot(x, data[:,0], x, data1[:,0])
ax1.legend(['x','x1'])
ax4 = fig.add_subplot(2,3,2)
ax4.plot(x, data[:,1], x, data1[:,1])
ax4.legend(['y','y1'])
ax5 = fig.add_subplot(2,3,3)
ax5.plot(x, data[:,2], x, data1[:,2])
ax5.legend(['z','z1'])


# 速度绘图：
length = data.shape[0]
x = np.linspace(0, length-1, length)
print(x)
data2 = ptsInWorld
dx = np.diff(data2[:,0]).T / deltaTime
dy = np.diff(data2[:,1]).T / deltaTime
dz = np.diff(data2[:,2]).T / deltaTime


ax2 = fig.add_subplot(2,3,4)
ax2.plot(x,dx,x,mu[:,1])
ax2.legend(['x','p'])
ax3 = fig.add_subplot(2,3,5)
ax3.plot(x,dy,x,mu[:,3])
ax3.legend('y','p')
ax4 = fig.add_subplot(2,3,6)
ax4.plot(x,dz,x,mu[:,5])
ax4.legend('z','p')


plt.savefig('./assets/ptsInWorld.jpg')


 
mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
#ax = Axes3D(fig)
x=data[:,0]
y=data[:,2]
z=data[:,1]*(-1)

x1=data1[:,0]
y1=data1[:,2]
z1=data1[:,1]*(-1)

ax.set_box_aspect((1,1,1))
ax.plot(x,y,z, label='ori')
ax.plot(x1,y1,z1, label='ekf')
ax.legend()

print(ax.get_xticks())
 
plt.savefig('./assets/test.jpg')
plt.show()


