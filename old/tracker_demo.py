import cv2
import math
import time
import numpy as np

import modules.tools as tools
from modules.autoaim.armor_detector import ArmorDetector
from modules.tracker import Tracker, TrackerState
from modules.NewEKF import ExtendedKalmanFilter
from modules.shot_point import Shot_Point
from modules.tracker import f

from remote_visualizer import Visualizer


class Target_msg:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.v_yaw = 0.0
        self.radius_1 = 0.0
        self.radius_2 = 0.0
        self.y_2 = 0.0


if __name__ == '__main__':
    with Visualizer() as visualizer:
        from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

        # video_path = 'assets/antitop_top.mp4'
        video_path = 'assets/edit.mp4'
        # video_path = 'assets/input.avi'

        cap = cv2.VideoCapture(video_path)
        armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        # EKF整车观测：
        # Tracker
        max_match_distance = 0.2  # 单位:m
        tracking_threshold = 5  # 从检测到->跟踪的帧数
        lost_threshold = 5  # 从暂时丢失->丢失的帧数
        tracker = Tracker(max_match_distance, tracking_threshold, lost_threshold)
        
        robot_yaw = -74.02
        robot_pitch = -0.37

        while True:
            success, frame = cap.read()
            if not success:
                break

            armors = armor_detector.detect(frame, robot_yaw, robot_pitch)
            armors.sort(key=lambda a: a.observation[0]) # 优先击打最近的

            if tracker.tracker_state == TrackerState.LOST:
                # 进入LOST状态后，必须要检测到装甲板才能初始化tracker
                if len(armors) == 0:
                    print('lost.')
                    continue

                tracker.init(armors)

            else:
                tracker.update(armors, 1/60)

            state = tracker.target_state

            msg = Target_msg()
            msg.position = [state[0], state[1], state[2]]
            msg.yaw = state[3]
            msg.velocity = [state[4], state[5], state[6]]
            msg.v_yaw = state[7]
            msg.radius_1 = state[8]
            msg.radius_2 = tracker.last_r
            msg.y_2 = tracker.last_y

            Shot = Shot_Point() 

            target_state = tracker.target_state # after filter   

            flyTime = tools.getParaTime(state[:3] * 1000, 17) / 1000      
        
            p_state = f(state, 0.05+flyTime) # predicted     
            p_state = np.reshape(p_state,(9,))    
            
            # 以下为调试相关的代码

            yaw_in_imu = 0  
            if len(armors)>0:
                yaw_in_imu = armors[0].yawR_in_imu           

            drawing = cv2.convertScaleAbs(frame, alpha=3)

            # 重投影
            R_imu2gimbal = tools.R_gimbal2imu(robot_yaw, robot_pitch).T
            R_gimbal2camera = R_camera2gimbal.T

            # 车辆中心
            center_in_imu = np.array(msg.position).reshape(3,1) * 1000
            center_in_gimbal = R_imu2gimbal @ center_in_imu
            center_in_camera = R_gimbal2camera @ center_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            center_in_pixel, _ = cv2.projectPoints(center_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
            center_in_pixel = center_in_pixel[0][0]
            tools.drawPoint(drawing, center_in_pixel, (0, 255, 255), radius=20)

            # 装甲板1
            armor1_in_imu = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000
            armor1_in_gimbal = R_imu2gimbal @ armor1_in_imu
            armor1_in_camera = R_gimbal2camera @ armor1_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor1_in_pixel, _ = cv2.projectPoints(armor1_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
            armor1_in_pixel = armor1_in_pixel[0][0]
            tools.drawPoint(drawing, armor1_in_pixel, (0, 0, 255), radius=10)

            # 装甲板2
            state = state.copy()
            state[1] = msg.y_2
            state[3] = msg.yaw + math.pi/2
            state[8] = msg.radius_2
            armor2_in_imu = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000
            armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
            armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
            armor2_in_pixel = armor2_in_pixel[0][0]
            tools.drawPoint(drawing, armor2_in_pixel, (0, 0, 255), radius=10)

            # 装甲板3
            state = state.copy()
            state[1] = msg.y_2
            state[3] = msg.yaw - math.pi/2
            state[8] = msg.radius_2
            armor3_in_imu = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000
            armor3_in_gimbal = R_imu2gimbal @ armor3_in_imu
            armor3_in_camera = R_gimbal2camera @ armor3_in_gimbal - R_gimbal2camera @ t_camera2gimbal
            armor3_in_pixel, _ = cv2.projectPoints(armor3_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
            armor3_in_pixel = armor3_in_pixel[0][0]
            tools.drawPoint(drawing, armor3_in_pixel, (0, 0, 255), radius=10)

            for a in armors:
                tools.drawContour(drawing, a.points)
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                x, y, z = a.in_imu.T[0]
                tools.putText(drawing, f'rotate speed: {msg.v_yaw:.2f}', (100, 100), (255, 255, 255))
                tools.putText(drawing, f'yaw_in_imu: {yaw_in_imu:.2f} yaw: {msg.yaw:.2f}', (100, 400), (255, 255, 255))
                tools.putText(drawing, f'radius1: {msg.radius_1:.2f} radius2: {msg.radius_2:.2f}', (100, 200), (255, 255, 255))
                tools.putText(drawing, f'x: {msg.position[0]:.2f} y: {msg.position[1]:.2f} z: {msg.position[2]:.2f}', (100, 300), (255, 255, 255))

            # cv2.imshow('press q to exit', drawing)
            visualizer.show(drawing)
        
            # visualizer.plot((yaw_in_imu, msg.yaw), ('yaw_in_imu','yaw'))
            # visualizer.plot((yaw_in_imu, msg.yaw, msg.v_yaw, 
            #                  msg.radius_1,msg.radius_2,
            #                  msg.position[0],msg.position[1],msg.position[2],
            #                  tracker.arrmor_jump, tracker.state_error,
            #                  int(tracker.tracker_state)), 
            #                  ('yaw_in_imu','yaw','v_yaw',
            #                   'r1','r2',
            #                   'x','y','z',
            #                   'arrmor_jump','state_error',
            #                   'tracker_state'))
            
            visualizer.plot((yaw_in_imu, msg.yaw, msg.v_yaw, 
                    msg.radius_1,msg.radius_2,
                    msg.position[0],msg.position[1],msg.position[2],p_state[0],
                    msg.velocity[0], msg.velocity[1], msg.velocity[2],
                    int(tracker.tracker_state)), 
                    ('yaw_in_imu','yaw','v_yaw',
                    'r1','r2',
                    'x','y','z','px',
                    'vx','vy','vz',
                    'tracker_state'))

            key = cv2.waitKey(16) & 0xff
            if key == ord('q'):
                break

        cap.release()
