import numpy as np

# 卡尔曼滤波器的参数Q和R矩阵
# ori:
#Q = [[56720.0,0.0,0.0],[0.0,316.0,0.0],[0.0,0.0,388.5]] # 对角线：状态值x,y,z方差
#Rr = [[99.96,0.0,0.0],[0.0,0.001948,0.0],[0.0,0.0,1.616]] # 对角线：相机测量值z、α、β方差

# test:
Q = np.float32([[86720000.0,0.0,0.0],[0.0,316000.0,0.0],[0.0,0.0,38800000.5]]) # 对角线：状态值x,y,z方差
Rr = np.float32([[99.96,0.0,0.0],[0.0,0.0001948,0.0],[0.0,0.0,0.000616]])

# zero:
#Q = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]
#Rr = [[0.0,0.0,0.0],[0.0,0.0,0.0],[0.0,0.0,0.0]]

#Q = [[500000.0,0.0,0.0],[0.0,3160.0,0.0],[0.0,0.0,388.5]] # 对角线：状态值x,y,z方差
#Rr = [[100.96,0.0,0.0],[0.0,0.00015,0.0],[0.0,0.0,0.616]] # 对角线：相机测量值z、α、β方差