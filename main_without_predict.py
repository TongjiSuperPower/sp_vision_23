import cv2
import time
import numpy as np
from collections import deque

import modules.tools as tools
from modules.io.robot import Robot
from modules.ExtendKF import EKF
from modules.armor_detection import ArmorDetector

from remote_visualizer import Visualizer

if __name__ == '__main__':
    with Robot(3, '/dev/ttyUSB0') as robot, Visualizer(enable=False) as visualizer:
        
        robot.update()
        if robot.id == 3:
            from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal
        elif robot.id == 4:
            from configs.infantry4 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

        armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        while True:
            robot.update()

            img = robot.img
            armors = armor_detector.detect(img, robot.yaw, robot.pitch)
            armors = list(filter(lambda a: a.color != robot.color, armors))
            armors.sort(key=lambda a: a.observation[0]) # 优先击打最近的

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
            visualizer.show(drawing)

            if len(armors) > 0:
                armor = armors[0]
                
                cx, cy, cz = armor.in_camera.T[0]
                x, y, z = armor.in_imu.T[0]
                imu_yaw = armor.yaw_in_imu

                robot.send(x, y, z)

                # 调试用
                visualizer.plot((imu_yaw, robot.yaw, robot.pitch), ('imu_yaw', 'yaw', 'pitch'))
