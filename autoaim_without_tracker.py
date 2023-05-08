import cv2
import math
import time

import modules.tools as tools
from modules.io.robot import Robot
from modules.autoaim.armor_detector import ArmorDetector
from modules.autoaim.armor_solver import ArmorSolver

from remote_visualizer import Visualizer


if __name__ == '__main__':
    enable: str = None
    while True:
        enable = input('开启Visualizer?输入[y/n]\n')
        if enable == 'y' or enable == 'n':
            break
        else:
            print('请重新输入')
    enable = True if enable == 'y' else False

    with Robot(5, '/dev/ttyUSB0') as robot, Visualizer(enable) as visualizer:
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
            armors = armor_solver.solve(armors, yaw_degree, pitch_degree)

            # 优先击打最近的
            armors = sorted(armors, key=lambda a: a.in_camera_mm[2])

            if len(armors) > 0:
                armor = armors[0]

                cx, cy, cz = armor.in_camera_mm.T[0]
                x, y, z = armor.in_imu_mm.T[0]

                send_pitch_degree = tools.shoot_pitch(x, y, z, robot.bullet_speed) + pitch_offset
                armor_in_gun_mm = armor.in_imu_mm.copy()
                armor_in_gun_mm[1] = (x*x + z*z) ** 0.5 * -math.tan(math.radians(send_pitch_degree))

                robot.send(*armor_in_gun_mm.T[0])

            # 调试用
            if not visualizer.enable:
                continue

            drawing = img.copy()
            for i, l in enumerate(armor_detector._raw_lightbars):
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)
            for i, lp in enumerate(armor_detector._raw_lightbar_pairs):
                tools.drawContour(drawing, lp.points)
            for i, a in enumerate(armors):
                cx, cy, cz = a.in_camera_mm.T[0]
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))

            visualizer.show(drawing)
