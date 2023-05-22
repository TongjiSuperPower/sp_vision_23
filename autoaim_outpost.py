import sys
import cv2
import math
import time
import logging
import numpy as np

import modules.tools as tools
from modules.io.robot import Robot
from modules.io.communication import Communicator
from modules.autoaim.armor_solver import ArmorSolver
from modules.autoaim.armor_detector import ArmorDetector, is_armor, is_lightbar, is_lightbar_pair
from modules.autoaim.tracker import Tracker

from remote_visualizer import Visualizer


logging.basicConfig(format='[%(asctime)s][%(levelname)s]%(message)s', level=logging.DEBUG)

exposure_ms = 3
port = '/dev/ttyUSB0'


if __name__ == '__main__':
    enable = False
    if len(sys.argv) > 1:
        enable = (sys.argv[1] == '-y')

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

        tracker = Tracker()

        while True:
            robot.update()

            img = robot.img
            img_time_s = robot.img_time_s

            armors = armor_detector.detect(img)

            yaw_degree, pitch_degree = robot.yaw_pitch_degree_at(img_time_s)
            armors = armor_solver.solve(armors, yaw_degree, pitch_degree)

            # print(f'Tracker state: {tracker.state} ')

            if tracker.state == 'LOST':
                tracker.init(armors, img_time_s)
            else:
                tracker.update(armors, img_time_s)

            if tracker.state in ('TRACKING', 'TEMP_LOST'):
                target = tracker.target
                aim_point_in_imu_m, fire_time_s = target.aim(robot.bullet_speed)

                aim_point_in_imu_mm = aim_point_in_imu_m * 1e3
                x, y, z = aim_point_in_imu_mm.T[0]
                send_pitch_degree = tools.shoot_pitch(x, y, z, robot.bullet_speed) + pitch_offset
                aim_point_in_imu_mm[1, 0] = (x*x + z*z) ** 0.5 * -math.tan(math.radians(send_pitch_degree))
                aim_point_in_imu_m = aim_point_in_imu_mm / 1e3
                
                robot.shoot(aim_point_in_imu_m, fire_time_s)

            # 调试分割线

            if not visualizer.enable:
                continue

            # drawing = img.copy()
            drawing = cv2.convertScaleAbs(img, alpha=5)

            for i, l in enumerate(armor_detector._raw_lightbars):
                if not is_lightbar(l):
                    continue
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)

            # for i, lp in enumerate(armor_detector._raw_lightbar_pairs):
            #     if not is_lightbar_pair(lp):
            #         continue
            #     tools.drawContour(drawing, lp.points, (0, 255, 255), 1)
            #     tools.putText(drawing, f'{lp.angle:.2f}', lp.left.top, (255, 255, 255))

            for i, a in enumerate(armor_detector._raw_armors):
                if not is_armor(a):
                    continue
                tools.drawContour(drawing, a.points)
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                
                # cx, cy, cz = a.in_camera_mm.T[0]
                # tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            if tracker.state != 'LOST':
                
                center_in_imu_m = tracker.target._ekf.x[:3]
                speed_rad_per_s = tracker.target._ekf.x[4, 0]
                center_in_imu_mm = center_in_imu_m * 1e3
                center_in_pixel = tools.project_imu2pixel(
                    center_in_imu_mm,
                    yaw_degree, pitch_degree,
                    cameraMatrix, distCoeffs,
                    R_camera2gimbal, t_camera2gimbal
                )
                tools.drawPoint(drawing, center_in_pixel, (0, 255, 255), radius=10)
                tools.putText(drawing, f'{speed_rad_per_s:.2f}', center_in_pixel, (255, 255, 255))

                x, y, z = center_in_imu_m.T[0]
                outpost_yaw_rad = tracker.target._ekf.x[3, 0]
                messured_yaw_rad = tracker.target.debug_yaw_rad
                outpost_yaw_degree = math.degrees(outpost_yaw_rad)
                robot_yaw_rad = math.radians(yaw_degree)

                # visualizer.plot((x, y, z), ('x', 'y', 'z'))
                visualizer.plot((speed_rad_per_s, outpost_yaw_rad, messured_yaw_rad, robot_yaw_rad), ('speed', 'yaw', 'm_yaw', 'robot_yaw'))

                for i, armor_in_imu_m in enumerate(tracker.target.get_all_armor_positions_m()):
                    armor_in_imu_mm = armor_in_imu_m * 1e3
                    armor_in_pixel = tools.project_imu2pixel(
                        armor_in_imu_mm,
                        yaw_degree, pitch_degree,
                        cameraMatrix, distCoeffs,
                        R_camera2gimbal, t_camera2gimbal
                    )
                    tools.drawPoint(drawing, armor_in_pixel, (0, 0, 255), radius=10)

            visualizer.show(drawing)
