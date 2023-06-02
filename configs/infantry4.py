import numpy as np

cameraMatrix = np.float32([[1650.5693178279687, 0.0, 642.1157815869858], [0.0, 1651.1493758734318, 520.8245852926152], [0.0, 0.0, 1.0]])
distCoeffs = np.float32([-0.49058344276059324, 0.35670287750830637, -0.0028462651720275673, 0.0010381728580505319, -0.3027436536427579])
R_camera2gimbal = np.float32([[0.9999941737974147, -0.0020017841241787813, 0.0027650011837269278], [0.002036793812128234, 0.9999170550928724, -0.012717484238813477], [-0.0027393141829117847, 0.01272304188147547, 0.9999153065950583]])
t_camera2gimbal = np.float32([[-0.4843357023401955], [62.636469719758225], [166.78727570110252]])
# 重投影误差: 0.0121px
# 相机相对于云台: yaw=0.16 pitch=0.73 roll=0.12

gun_up_degree = -2
gun_right_degree = 0

whitelist = ('small_two',)
