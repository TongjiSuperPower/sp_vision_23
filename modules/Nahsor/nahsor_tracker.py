from modules.Nahsor.nahsor_marker import NahsorMarker
import configs.NahsorConfig as NahsorConfig
import modules.tools as tools
from modules.Nahsor.nahsor_solver import NahsorSolver
import numpy as np

class NahsorTracker():
    '''能量机关追踪器'''
    def __init__(self,robot_color) -> None:
        self.nahsor:NahsorMarker = None 
        self.nahsor_color = NahsorConfig.COLOR.BLUE if robot_color == 'red' else NahsorConfig.COLOR.RED 
        self.last_mode = 0
        self.color_space = NahsorConfig.COLOR_SPACE.HSV if robot_color == 'red' else NahsorConfig.COLOR_SPACE.YUV
        self.init(False)     

    def init(self, isBig):    
        if isBig:    
            energy_mode=NahsorConfig.ENERGY_MODE.BIG
            
        else:
            energy_mode=NahsorConfig.ENERGY_MODE.SMALL

        self.nahsor = NahsorMarker(color=self.nahsor_color, energy_mode = energy_mode,
                                   color_space=self.color_space,
                                   fit_debug=0, target_debug=1,
                                   fit_speed_mode=NahsorConfig.FIT_SPEED_MODE.CURVE_FIT)


    def update(self, frame, robot_work_mode):
        '''用图像帧更新,进行: 风车轮廓识别识别,叶片识别,R标拟合,速度估计'''
        # 根据robot工作模式重新初始化风车对象
        if robot_work_mode != self.last_mode:
            if robot_work_mode == 2:
                self.init(False)
            else:
                self.init(True)
            self.last_mode = robot_work_mode
        
        self.nahsor.mark(frame)

    def getShotPoint(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw_degree=0, pitch_degree=0, enablePredict = NahsorConfig.USE_PREDICT):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        # 如果没有找到风车,就返回None
        if self.nahsor.getTargetStatus() == NahsorConfig.STATUS.NOT_FOUND:
            print("lost nahsor")
            return None
        
        # 观测到的靶心坐标
        nahsorSolver = NahsorSolver(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)
        
        current2DCorners = self.nahsor.target_corners
        if current2DCorners is None:
            # print("target corners is None")
            return None
        
        # current2DCorners = np.vstack((current2DCorners, self.nahsor.r_center)).astype(np.float32) # 图像坐标系中的5个角点的xy坐标
        current2DCorners = current2DCorners.astype(np.float32)
        current3DPos = nahsorSolver.solve(points_2d=current2DCorners, 
                                          yaw_degree=yaw_degree, 
                                          pitch_degree=pitch_degree).in_imu_mm # pnp解算出观测靶心的世界坐标系3d坐标
        
        # print(current2DCorners)

        if not enablePredict:
            return current3DPos
        
        # 预测一段时间后的靶心坐标

        # 如果速度拟合还没有完成,就不进行预测,返回None
        if self.nahsor.fit_status is not NahsorConfig.FIT_STATUS.SUCCESS:            
            return None
            
        flyTime_s = tools.getParaTime(current3DPos, bulletSpeed=bulletSpeed) / 1000 # 到观测靶心的子弹飞行时间(秒)(s)

        predictTime_s = flyTime_s + deltatime # 预测时间

        predict2DCorners = self.nahsor.get_2d_predict_corners(predict_time=predictTime_s)
        # predict2DCorners = np.vstack((predict2DCorners, self.nahsor.r_center)).astype(np.float32) # 预测时间后图像坐标系中的5个角点的xy坐标
        predict2DCorners = predict2DCorners.astype(np.float32)
        
        predict3DPos = nahsorSolver.solve(points_2d=predict2DCorners, 
                                          yaw_degree=yaw_degree, 
                                          pitch_degree=pitch_degree).in_imu_mm # pnp解算出观测靶心的世界坐标系3d坐标

        return predict3DPos

