import cv2
import time
import numpy as np
from collections import deque
import math

import modules.tools as tools
from modules.io.robot import Robot

from modules.ExtendKF import EKF
from modules.armor_detection import ArmorDetector
import modules.tools as tools
from modules.tracker import Tracker, TrackerState
from modules.NewEKF import ExtendedKalmanFilter
from modules.shot_point import Shot_Point

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
    with Robot(3, '/dev/ttyUSB0') as robot, Visualizer(enable=False) as visualizer:
        
        robot.update()
        if robot.id == 3:
            from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal
        elif robot.id == 4:
            from configs.infantry4 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

        armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        ekfilter = EKF(6, 3)
        maxLostFrame = 3  # 最大丢失帧数
        lostFrame = 0  # 丢失帧数
        state = np.zeros((6, 1))
        twoPtsInCam = deque(maxlen=2)  # a queue with max 2 capaticity
        twoPtsInWorld = deque(maxlen=2)
        twoPtsInTripod = deque(maxlen=2)
        twoTimeStampMs = deque(maxlen=2)
        continuousFrameCount = 0

        while True:
            robot.update()
            img = robot.img
            
            twoTimeStampMs.append(robot.camera_stamp_ms)
            dtMs = (twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 5 # (ms)
            dt = dtMs/1000
            
            Shot = Shot_Point() 
            armors = armor_detector.detect(img, robot.yaw, robot.pitch)
            armors = list(filter(lambda a: a.color != robot.color, armors))
            """sentry 五米以内优先选英雄        其次打步兵       """
            armors.sort(key=lambda a: a.observation[0]) # 优先击打最近的
            armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

            # EKF整车观测：
            # Tracker
            max_match_distance = 0.2  # 单位:m
            tracking_threshold = 5  # 从检测到->跟踪的帧数
            lost_threshold = 5  # 从暂时丢失->丢失的帧数
            tracker = Tracker(max_match_distance, tracking_threshold, lost_threshold)
            
            
            # 调试用
            drawing = img.copy()
            for i, l in enumerate(armor_detector._filtered_lightbars):
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)
            for i, lp in enumerate(armor_detector._filtered_lightbar_pairs):
                tools.drawContour(drawing, lp.points)
            for i, a in enumerate(armors):
                cx, cy, cz = a.in_camera.T[0]
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))
            

            if tracker.tracker_state == TrackerState.LOST:
                # 进入LOST状态后，必须要检测到装甲板才能初始化tracker
                if len(armors) == 0:
                    print('lost.')
                    continue

                tracker.init(armors)

            else:    
                tracker.update(armors, dt)
                target_state = tracker.target_state # after filter            
                
                # ___________________________________________________________________________________________________
                msg = Target_msg()
                msg.position = [target_state[0], target_state[1], target_state[2]]
                msg.yaw = target_state[3]
                msg.velocity = [target_state[4], target_state[5], target_state[6]]
                msg.v_yaw = target_state[7]
                msg.radius_1 = target_state[8]
                msg.radius_2 = tracker.last_r
                msg.y_2 = tracker.last_y
                # 重投影
                R_imu2gimbal = tools.R_gimbal2imu(0, 0).T
                R_gimbal2camera = R_camera2gimbal.T

                # 车辆中心
                center_in_imu = np.array(msg.position).reshape(3,1) * 1000
                center_in_gimbal = R_imu2gimbal @ center_in_imu
                center_in_camera = R_gimbal2camera @ center_in_gimbal - R_gimbal2camera @ t_camera2gimbal
                center_in_pixel, _ = cv2.projectPoints(center_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
                center_in_pixel = center_in_pixel[0][0]
                tools.drawPoint(drawing, center_in_pixel, (0, 255, 255), radius=20)

                # 装甲板1
                armor1_in_imu = np.array(tracker.getArmorPositionFromState(target_state)).reshape(3, 1) * 1000
                armor1_in_gimbal = R_imu2gimbal @ armor1_in_imu
                armor1_in_camera = R_gimbal2camera @ armor1_in_gimbal - R_gimbal2camera @ t_camera2gimbal
                armor1_in_pixel, _ = cv2.projectPoints(armor1_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
                armor1_in_pixel = armor1_in_pixel[0][0]
                tools.drawPoint(drawing, armor1_in_pixel, (0, 0, 0), radius=10)#bgr_black 滤波之后四块装甲板的位置

                # 装甲板2
                state = target_state.copy()
                state[1] = msg.y_2
                state[3] = msg.yaw + math.pi/2
                state[8] = msg.radius_2
                armor2_in_imu = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000
                armor2_in_gimbal = R_imu2gimbal @ armor2_in_imu
                armor2_in_camera = R_gimbal2camera @ armor2_in_gimbal - R_gimbal2camera @ t_camera2gimbal
                armor2_in_pixel, _ = cv2.projectPoints(armor2_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
                armor2_in_pixel = armor2_in_pixel[0][0]
                tools.drawPoint(drawing, armor2_in_pixel, (0, 0, 0), radius=10)

                # 装甲板3
                state = target_state.copy()
                state[1] = msg.y_2
                state[3] = msg.yaw - math.pi/2
                state[8] = msg.radius_2
                armor3_in_imu = np.array(tracker.getArmorPositionFromState(state)).reshape(3, 1) * 1000
                armor3_in_gimbal = R_imu2gimbal @ armor3_in_imu
                armor3_in_camera = R_gimbal2camera @ armor3_in_gimbal - R_gimbal2camera @ t_camera2gimbal
                armor3_in_pixel, _ = cv2.projectPoints(armor3_in_camera, np.zeros((3,1)), np.zeros((3,1)), cameraMatrix, distCoeffs)
                armor3_in_pixel = armor3_in_pixel[0][0]
                tools.drawPoint(drawing, armor3_in_pixel, (0, 0, 0), radius=10)
                # ___________________________________________________________________________________________________
                
                
                predictedPtsInWorld = Shot.get_predicted_shot_point(target_state, tracker, 0.05, robot.bullet_speed, 1)
                 
                tools.drawPoint(drawing, Shot.shot_point_in_pixel,(0,0,255),radius = 10)#red 预测时间后待击打装甲板的位置
                
                cx, cy, cz = tracker.tracked_armor.in_camera.T[0]
                x, y, z = tracker.tracked_armor.in_imu.T[0]
                px, py, pz = target_state.T[0]
                ppx, ppy, ppz = predictedPtsInWorld


                # robot.send(x, y, z)
                # robot.send(px, py, pz)
                if robot.id == 0:
                    if target_state[2]<5e3:# sentry Within five meters
                        robot.send(ppx, ppy-60, ppz)
                else:
                    robot.send(ppx, ppy-60, ppz)
               
               
                visualizer.show(drawing)
                # 调试用
                # visualizer.plot((cy, y, robot.yaw*10, robot.pitch*10), ('cy', 'y', 'yaw', 'pitch'))
                visualizer.plot((x, y, z, robot.yaw*10, robot.pitch*10), ('x', 'y', 'z', 'yaw', 'pitch'))
                # visualizer.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz,vx*10,vy*10,vz*10), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz','vx','vy','vz'))
