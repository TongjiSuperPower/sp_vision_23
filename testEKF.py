# 测试基于扩展卡尔曼滤波（EKF）的装甲板运动预测功能
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt
import math
from modules.ExtendKF import EKF

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

ptsEKF = np.zeros((ptsInWorld.shape))
ekf = EKF(6, 3)
state = np.zeros((6,1))
all_z = []
all_alpha = []
all_beta = []
pdx=[]
pdy=[]
pdz=[]
prePts = []

for i in range(ptsInWorld.shape[0]):
    x = ptsInCam[i,0]
    y = ptsInCam[i,1]
    z = ptsInCam[i,2]
    alpha = math.atan(-x/z)
    beta = math.atan(-y/z)
    observation = [z, alpha, beta]

    all_z.append(z)
    all_alpha.append(alpha)
    all_beta.append(beta)
    #print(observation)

    deltaTime = 5 # ms

    if ekf.first==False:
        state[1,0] = (ptsInWorld[i,0] - ptsInWorld[i-1,0])/deltaTime
        state[3,0] = (ptsInWorld[i,1] - ptsInWorld[i-1,1])/deltaTime
        state[5,0] = (ptsInWorld[i,2] - ptsInWorld[i-1,2])/deltaTime

        # state[1,0] = 0
        # state[3,0] = 0
        # state[5,0] = 0
        
    
    state[0,0] = ptsInWorld[i,0]
    state[2,0] = ptsInWorld[i,1]
    state[4,0] = ptsInWorld[i,2]

    #print(state)    

    predictedPtsInWorld = ekf.step(deltaTime, [yaw,pitch], state, observation, np.reshape(ptsInCam[i], (3,1)))
    ptsEKF[i] = predictedPtsInWorld.T

    distance = np.linalg.norm(ekf.hMatrix @ ekf.state) # 世界坐标系下的距离(mm)
    flyTime = distance/15.0
    prePts.append(ekf.getPredictedPtsInWorld(5+flyTime))

    pdx.append(ekf.state[1][0])
    pdy.append(ekf.state[3][0])
    pdz.append(ekf.state[5][0])

prePts = np.array(prePts)
data = ptsInWorld
data1 = ptsEKF

# print('世界坐标系x、y、z的方差：')
# print(np.var(ptsInWorld[:,0]))
# print(np.var(ptsInWorld[:,1]))
# print(np.var(ptsInWorld[:,2]))
# print('z、alpha、beta的方差：')
# print(np.var(all_z))
# print(np.var(all_alpha))
# print(np.var(all_beta))

length = data.shape[0]

x = np.linspace(0, length-1, length)
fig = plt.figure()
ax1 = fig.add_subplot(2,3,1)
ax1.plot(x, data[:,0])
ax1.plot(x, data1[:,0])
ax1.plot(x, prePts[:,0])
ax1.legend(['x','x1'])
ax4 = fig.add_subplot(2,3,2)
ax4.plot(x, data[:,1], x, data1[:,1], x, prePts[:,1])
ax4.legend(['y','y1'])
ax5 = fig.add_subplot(2,3,3)
ax5.plot(x, data[:,2], x, data1[:,2], x, prePts[:,2])
ax5.legend(['z','z1'])


# 速度绘图：
length = data.shape[0]-1
x = np.linspace(0, length-1, length)
print(x)
dx = np.diff(data[:,0]).T / deltaTime
dy = np.diff(data[:,1]).T / deltaTime
dz = np.diff(data[:,2]).T / deltaTime

ax2 = fig.add_subplot(2,3,4)
ax2.plot(x,dx,x,pdx[1:])
ax2.legend(['x','p'])
ax3 = fig.add_subplot(2,3,5)
ax3.plot(x,dy,x,pdy[1:])
ax3.legend('y','p')
ax4 = fig.add_subplot(2,3,6)
ax4.plot(x,dz,x,pdz[1:])
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


