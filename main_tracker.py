import cv2
import time
import numpy as np
from collections import deque
import math

import modules.tools as tools
from modules.io.robot import Robot
from modules.io.recording import Recorder
from modules.io.communication import Communicator

from modules.armor_detection import ArmorDetector
import modules.tools as tools
from modules.tracker import Tracker, TrackerState

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
    port = '/dev/ttyUSB0'
    with Communicator(port) as communicator:
        communicator.receive_no_wait(True)

    with Robot(3, port) as robot, Visualizer(enable=True) as visualizer, Recorder() as recorder:
        
        robot.update()
        if robot.id == 3:
            from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset
        elif robot.id == 4:
            from configs.infantry4 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset
        elif robot.id == 7:
            from configs.sentry import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset

        armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        twoTimeStampMs = deque(maxlen=2)

        # EKF整车观测：
        # Tracker
        max_match_distance = 0.2  # 单位:m
        tracking_threshold = 5  # 从检测到->跟踪的帧数
        lost_threshold = 10  # 从暂时丢失->丢失的帧数
        tracker = Tracker(max_match_distance, tracking_threshold, lost_threshold)

        while True:
            robot.update()
            img = robot.img
            robot_yaw, robot_pitch = robot.yaw_pitch_degree_at(robot.img_time_s)
            robot_stamp_ms = robot.img_time_s * 1e3
            recorder.record(img, (robot_stamp_ms, robot_yaw, robot_pitch))
            
            twoTimeStampMs.append(robot_stamp_ms)
            dtMs = (twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 8 # (ms)
            dt = dtMs/1000
            
            armors = armor_detector.detect(img, robot_yaw, robot_pitch)
            armors = list(filter(lambda a: a.color != robot.color, armors))
            armors.sort(key=lambda a: a.observation[0]) # 优先击打最近的
            
            # 调试用
            drawing = cv2.convertScaleAbs(img, alpha=10)
            for i, l in enumerate(armor_detector._filtered_lightbars):
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)
            for i, lp in enumerate(armor_detector._filtered_lightbar_pairs):
                tools.drawContour(drawing, lp.points)
            for i, a in enumerate(armor_detector._armors):
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
            for i, a in enumerate(armors):
                cx, cy, cz = a.in_camera.T[0]
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            visualizer.show(drawing)

            if tracker.tracker_state == TrackerState.LOST:

                tracker.init(armors)

            else:    
                tracker.update(armors, dt)                          
                
                predictedPtsInWorld = tracker.getShotPoint(0.05, robot.bullet_speed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, robot_yaw, robot_pitch)
                 
                # tools.drawPoint(drawing, Shot.shot_point_in_pixel,(0,0,255),radius = 10)#red 预测时间后待击打装甲板的位置

                # 重力补偿                
                armor_in_gun = tools.trajectoryAdjust(predictedPtsInWorld, pitch_offset, robot.bullet_speed
                                                      , enableAirRes=1)
                
                fire = 1 if tracker.tracker_state == TrackerState.TRACKING else 0
                robot.send(*armor_in_gun.T[0], flag=fire)
           
                # 调试用
                # visualizer.plot((cy, y, robot_yaw*10, robot_pitch*10), ('cy', 'y', 'yaw', 'pitch'))
                # visualizer.plot((x, y, z, robot_yaw*10, robot_pitch*10), ('x', 'y', 'z', 'yaw', 'pitch'))
                # visualizer.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz,vx*10,vy*10,vz*10), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz','vx','vy','vz'))
