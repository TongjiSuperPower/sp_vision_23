import numpy as np

cameraMatrix = np.float32([[1650.734889432613, 0.0, 623.5014475324386], [0.0, 1651.2465097224272, 522.2186364540044], [0.0, 0.0, 1.0]])
distCoeffs = np.float32([-0.4846395510462699, 0.31047909784652583, -0.001641514962258196, -0.00011824465016236522, -0.1996546202822333])
R_camera2gimbal = np.float32([[0.9998032758670008, -0.019404105757786113, -0.00410977436785875], [0.019401055933792922, 0.9998114770243538, -0.0007806657632873763], [0.004124147701997997, 0.0007007782251058352, 0.9999912501195254]])
t_camera2gimbal = np.float32([[0.5778272722927769], [62.97059210358873], [166.29747178468284]])
# 重投影误差: 0.0099px
# 相机相对于云台: yaw=-0.24 pitch=0.04 roll=1.11

pitch_offset = -1
