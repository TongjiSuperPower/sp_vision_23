# 测试基于扩展卡尔曼滤波（EKF）的装甲板运动预测功能
import matplotlib as mpl
from mpl_toolkits.mplot3d import Axes3D
import numpy as np
import matplotlib.pyplot as plt

filePath = './assets/ptsInCam.txt'

data = np.loadtxt(filePath)

length = data.shape[0]

x = np.linspace(0, length-1, length)

plt.plot(x, data[:,0], x, data[:,1], x, data[:,2])
plt.legend(['x','y','z'])
plt.savefig('./assets/ptsInCam.jpg')


 
mpl.rcParams['legend.fontsize'] = 10
 
fig = plt.figure()
ax = fig.add_subplot(projection = '3d')
x=data[:,0]
y=data[:,2]
z=data[:,1]*(-1)

ax.plot(x[:100], y[:100], z[:100], label='parametric curve')
ax.legend()
 
plt.savefig('./assets/test.jpg')
plt.show()


