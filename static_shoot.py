import sys
import cv2
import math
import time
import logging
import numpy as np

import modules.tools as tools
from modules.io.robot import Robot
from modules.io.recorder import Recorder
from modules.io.communication import Communicator
from modules.autoaim.armor_solver import ArmorSolver
from modules.autoaim.armor_detector import ArmorDetector, is_armor, is_lightbar, is_lightbar_pair
from modules.autoaim.tracker import Tracker

from remote_visualizer import Visualizer


exposure_ms = 3
port = '/dev/ttyUSB0'


if __name__ == '__main__':
    tools.config_logging()
    
    enable = False
    if len(sys.argv) > 1:
        enable = (sys.argv[1] == '-y')

    with Communicator(port):
        # 这里的作用是在程序正式运行前，打开串口再关闭。
        # 因为每次开机后第一次打开串口，其输出全都是0，原因未知。
        pass

    with Robot(exposure_ms, port) as robot, Visualizer(enable=enable) as visualizer, Recorder() as recorder:
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

        while True:
            robot.update()

            img = robot.img
            img_time_s = robot.img_time_s

            armors = armor_detector.detect(img)

            yaw_degree, pitch_degree = robot.yaw_pitch_degree_at(img_time_s)

            recorder.record(img, (img_time_s, yaw_degree, pitch_degree, robot.bullet_speed, robot.flag))

            armors = armor_solver.solve(armors, yaw_degree, pitch_degree)
            
            armors = sorted(armors, key=lambda a: a.in_imu_mm[2, 0])

            if len(armors) > 0:
                armor = armors[0]

                aim_point_m = armor.in_imu_m

                x, y, z = (aim_point_m * 1e3).T[0]
                shoot_pitch_degree = tools.shoot_pitch(x, y, z, robot.bullet_speed)
                shoot_y_mm = (x*x + z*z) ** 0.5 * -math.tan(math.radians(shoot_pitch_degree))

                aim_point_m[1, 0] = shoot_y_mm / 1e3

                robot.shoot(pitch_offset, aim_point_m)

            # 调试分割线

            if not visualizer.enable:
                continue

            drawing = img.copy()
            # drawing = cv2.convertScaleAbs(img, alpha=5)

            for i, l in enumerate(armor_detector._raw_lightbars):
                if not is_lightbar(l):
                    continue
                tools.drawContour(drawing, l.points, (0, 0, 255), 1)

            # for i, lp in enumerate(armor_detector._raw_lightbar_pairs):
            #     if not is_lightbar_pair(lp):
            #         continue
            #     tools.drawContour(drawing, lp.points, (0, 255, 255), 1)
            #     tools.putText(drawing, f'{lp.angle:.2f}', lp.left.top, (255, 255, 255))

            for i, a in enumerate(armor_detector._raw_armors):
                if not is_armor(a):
                    continue
                # tools.drawContour(drawing, a.points)
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                
                # cx, cy, cz = a.in_camera_mm.T[0]
                # tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            if len(armors) > 0:
                armor = armors[0]
                # visualizer.plot((armor.in_camera_mm[0, 0]/10, armor.yaw_in_camera_degree,), ('x', 'yaw', ))

            # visualizer.show(armor_detector._gray_img)
            visualizer.show(drawing)