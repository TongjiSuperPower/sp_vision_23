import cv2
import time
import math
import numpy as np
from collections import deque

import modules.tools as tools
from modules.io.robot import Robot

from modules.autoaim.armor_detector import ArmorDetector
from modules.autoaim.armor_solver import ArmorSolver
import modules.tools as tools
from modules.tracker import Tracker, TrackerState
from modules.Nahsor.nahsor_tracker import NahsorTracker 

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
    enable: str = None
    while True:
        enable = input('开启Visualizer?输入[y/n]\n')
        if enable == 'y' or enable == 'n':
            break
        else:
            print('请重新输入')
    enable = True if enable == 'y' else False

    with Robot(5, '/dev/ttyUSB0') as robot, Visualizer(enable=enable) as visualizer:
        robot.update()

        if robot.id == 1:
            from configs.hero import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset
        elif robot.id == 3:
            from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset
        elif robot.id == 4:
            from configs.infantry4 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset
        elif robot.id == 7:
            from configs.sentry import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal, pitch_offset

        enemy_color = 'red' if robot.color == 'blue' else 'blue'
        armor_detector = ArmorDetector(enemy_color)
        armor_solver = ArmorSolver(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        twoTimeStampMs = deque(maxlen=2)

        # EKF整车观测：
        # Tracker
        max_match_distance = 0.2  # 单位:m
        tracking_threshold = 5  # 从检测到->跟踪的帧数
        lost_threshold = 10  # 从暂时丢失->丢失的帧数
        tracker = Tracker(max_match_distance, tracking_threshold, lost_threshold)
        
        # 反能量机关:
        nahsor_tracker = NahsorTracker(robot_color=robot.color)

        while True:
            robot.update()

            img = robot.img
            drawing = img.copy()

            img_time_s = robot.img_time_s
            robot_stamp_ms = robot.img_time_s * 1e3

            armors = armor_detector.detect(img)

            robot_yaw_degree, robot_pitch_degree = robot.yaw_pitch_degree_at(robot.img_time_s)
            armors = armor_solver.solve(armors, robot_yaw_degree, robot_pitch_degree)

            # 优先击打最近的
            armors = sorted(armors, key=lambda a: a.in_camera_mm[2])

            twoTimeStampMs.append(robot_stamp_ms)
            dtMs = (twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 8  # (ms)
            dt = dtMs/1000

            if robot.work_mode == 1:
                # 自瞄
                if tracker.tracker_state == TrackerState.LOST:

                    tracker.init(armors)

                else:
                    tracker.update(armors, dt)

                    predictedPtsInWorld = tracker.getShotPoint(0.05, robot.bullet_speed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, robot_yaw_degree, robot_pitch_degree)

                    # tools.drawPoint(drawing, Shot.shot_point_in_pixel,(0,0,255),radius = 10)#red 预测时间后待击打装甲板的位置

                    # 重力补偿
                    armor_in_gun = tools.trajectoryAdjust(predictedPtsInWorld, pitch_offset, robot, enableAirRes=1)

                    fire = 1 if tracker.tracker_state == TrackerState.TRACKING else 0
                    robot.send(*armor_in_gun.T[0], flag=fire)

                    # 调试用
                    # visualizer.plot((cy, y, robot_yaw_degree*10, robot_pitch_degree*10), ('cy', 'y', 'yaw', 'pitch'))
                    # visualizer.plot((x, y, z, robot_yaw_degree*10, robot_pitch_degree*10), ('x', 'y', 'z', 'yaw', 'pitch'))
                    # visualizer.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
                    # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz'))
                    visualizer.plot(
                        (tracker.tracking_target.target_state[0],tracker.tracking_target.target_state[1],tracker.tracking_target.target_state[2],
                         tracker.tracking_target.target_state[3],
                         tracker.tracking_target.target_state[4],tracker.tracking_target.target_state[5],tracker.tracking_target.target_state[6],
                         tracker.tracking_target.target_state[7],tracker.tracking_target.target_state[8]),
                         ('x','y','z','yaw','vx','vy','vz','vyaw','r')
                    )

                for i, l in enumerate(armor_detector._raw_lightbars):
                    tools.drawContour(drawing, l.points, (0, 255, 255), 10)
                for i, lp in enumerate(armor_detector._raw_armors):
                    tools.drawContour(drawing, lp.points)
                for i, a in enumerate(armors):
                    cx, cy, cz = a.in_camera_mm.T[0]
                    tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                    tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                    tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            
            else:
                # 能量机关
                nahsor_tracker.update(frame=img)

                predictedPtsInWorld = nahsor_tracker.getShotPoint(0.05, robot.bullet_speed, 
                                                                  R_camera2gimbal, t_camera2gimbal, 
                                                                  cameraMatrix, distCoeffs, 
                                                                  robot_yaw_degree, robot_pitch_degree, 
                                                                  enablePredict=0)

                if predictedPtsInWorld is not None:
                    armor_in_gun = tools.trajectoryAdjust(predictedPtsInWorld, pitch_offset, robot, enableAirRes=1)
                    # print(armor_in_gun)

                    robot.shoot(armor_in_gun/1000)


            # 调试用
            if not visualizer.enable:
                continue

            
            visualizer.show(drawing)
