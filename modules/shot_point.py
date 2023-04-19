import cv2
import math
import time
import numpy as np
from modules.NewEKF import ExtendedKalmanFilter
from modules.armor_detection import Armor
from modules.tracker import Tracker, TrackerState
import modules.tools as tools
from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

from modules.tracker import f

class Target_msg:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.v_yaw = 0.0
        self.radius_1 = 0.0
        self.radius_2 = 0.0
        self.y_2 = 0.0

class Shot_Point:
    def __init__(self) -> None:
        self.four_predict_points = None
        
        self.pre_armor_0 = None#正对着的那块（距离最近的）
        self.pre_armor_1 = None#以敌方车辆为中心顺时针90旋转的点
        self.pre_armor_2 = None#.....逆时针旋转90
        self.pre_armor_3 = None#....顺时针旋转180
        
        self.armors_in_pixel = []#重投影的像素点
        self.pre_armors_angle = []
        self.shot_point_in_imu = None
        self.shot_point_in_pixel = None
          
    
    def get_predicted_shot_point(self, state, tracker: Tracker, deltatime, bulletSpeed, enableGravity = 1):
        '''
        deltaTime: system delta time(s)
        '''    
        
        
        flyTime = tools.getParaTime(state*1000, bulletSpeed)        
        
        state = f(state,deltatime+flyTime) # predicted
        
        self.pre_armor_0 = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000# x y z
        
        _state = state.copy()
        _state[1] = tracker.last_y
        _state[3] = state[3]+ math.pi/2
        _state[8] = tracker.last_r
        self.pre_armor_1 = np.array(tracker.getArmorPositionFromState(_state)).reshape(3, 1) * 1000
        
        _state = state.copy()
        _state[1] = tracker.last_y
        _state[3] = state[3] - math.pi/2
        _state[8] = tracker.last_r
        self.pre_armor_2 = np.array(tracker.getArmorPositionFromState(_state)).reshape(3, 1) * 1000            
        
        # _state = state.copy()
        # _state[1] = tracker.last_y
        # _state[3] = state[3] + math.pi
        # _state[8] = tracker.last_r
        # self.pre_armor_3 = np.array(tracker.getArmorPositionFromState(_state)).reshape(3, 1) * 1000               
        
        self.four_predict_points = [self.pre_armor_0,self.pre_armor_1,self.pre_armor_2]
        # print("aaaa{}".format(self.four_predict_points))
        
                    
        # 重投影
        R_imu2gimbal = tools.R_gimbal2imu(0, 0).T
        R_gimbal2camera = R_camera2gimbal.T
        
        # 得到三个可疑点的重投影点 armors_in_pixel 
        # 与枪管的夹角
        for armor_state in self.four_predict_points:
            armor2_in_imu = armor_state
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            self.armors_in_pixel.append(armor2_in_pixel)
            
            # 注意单位，单位为mm
            a = (armor_state[0]**2 + armor_state[2]**2) 
            a = a[0]
            a = math.sqrt(a)
            b = math.sqrt((state[0]*1000)**2 + (state[2]*1000)**2) #
            c = tracker.last_r * 1000
            
            t = 0
            if self.is_triangle(a,b,c):
                if len(self.pre_armors_angle) != 0 :
                    for i in self.pre_armors_angle:
                        temp = self.triangle_angles(a , b, c)                        
                        if temp < t :
                            self.shot_point = armor2_in_pixel
                            
                else: #when len()=0 对列表初始化 以便后续排序                              
                    self.pre_armors_angle.append(self.triangle_angles(a , b, c))
                    t = self.triangle_angles(a , b, c)
                    self.shot_point_in_pixel = armor2_in_pixel
                    self.shot_point_in_imu = armor_state

            else:
                self.shot_point_in_pixel = armor2_in_pixel
                self.shot_point_in_imu = armor_state
                temp = 0
                self.pre_armors_angle.append(temp)
            
            if enableGravity:
                dropDistance = 0.5 * 9.7940 * flyTime**2
                self.shot_point_in_imu[1] -= dropDistance # 因为y轴方向向下，所以是减法 gravity
                
        return self.shot_point_in_imu

            

    def is_triangle(self, a, b, c):
        """
        Args:
            xo 自己车中心
            xa 敌方装甲板中心
            xc 敌方车中心
            a (xo 2 xa): 
            b (xo 2 xc): 
            c (xa 2 xc): 
        """
        if a + b > c and a + c > b and b + c > a:
            return True
        else:
            return False
        

    def triangle_angles(self, a, b, c):
        # 使用余弦定理计算角度
        angle_B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
        return (180 - angle_B)
    

