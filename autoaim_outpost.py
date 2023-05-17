import cv2
import math
import time

import modules.tools as tools
from modules.io.robot import Robot
from modules.io.communication import Communicator
from modules.autoaim.armor_solver import ArmorSolver
from modules.autoaim.armor_detector import ArmorDetector, is_armor, is_lightbar, is_lightbar_pair
from modules.autoaim.tracker import Tracker

from remote_visualizer import Visualizer


exposure_ms = 5
port = '/dev/ttyUSB0'
max_match_distance_m = 0.1
max_lost_count = 100
min_detect_count = 3 

if __name__ == '__main__':
    enable: str = None
    while True:
        enable = input('开启Visualizer?输入[y/n]\n')
        if enable == 'y' or enable == 'n':
            break
        else:
            print('请重新输入')
    enable = True if enable == 'y' else False

    with Communicator(port):
        # 这里的作用是在程序正式运行前，打开串口再关闭。
        # 因为每次开机后第一次打开串口，其输出全都是0，原因未知。
        pass

    with Robot(exposure_ms, port) as robot, Visualizer(enable=enable) as visualizer:
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
        
        tracker =  Tracker(max_match_distance_m, max_lost_count, min_detect_count)

        while True:
            robot.update()

            img = robot.img
            img_time_s = robot.img_time_s

            armors = armor_detector.detect(img)

            yaw_degree, pitch_degree = robot.yaw_pitch_degree_at(img_time_s)
            armors = armor_solver.solve(armors, yaw_degree, pitch_degree)

            print(f'Tracker state: {tracker.state} ')

            if tracker.state == 'LOST':
                tracker.init(armors, img_time_s)
            else:
                tracker.update(armors, img_time_s)

            # 调试分割线

            if not visualizer.enable:
                continue

            drawing = img.copy()

            for i, l in enumerate(armor_detector._raw_lightbars):
                if not is_lightbar(l):
                    continue
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)

            for i, a in enumerate(armor_detector._raw_armors):
                if not is_armor(a):
                    continue
                cx, cy, cz = a.in_camera_mm.T[0]
                tools.drawContour(drawing, a.points)
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            visualizer.show(drawing)
