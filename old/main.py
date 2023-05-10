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

            if len(armors) == 0:
                lostFrame += 1
                if lostFrame > maxLostFrame:
                    # after losing armor for a while
                    lostFrame = 0
                    ekfilter = EKF(6, 3)  # create a new filter
                    if continuousFrameCount > 0:
                        print(f'last {continuousFrameCount=}')
                    continuousFrameCount = 0

            if len(armors) > 0:
                continuousFrameCount += 1

                twoTimeStampMs.append(robot.camera_stamp_ms)
                deltaTime = (twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 5

                armor = armors[0]
                print(armor.yaw_in_imu)
                predictedPtsInWorld = armor.in_imu.T[0]  # 如果没开EKF，就发送识别值

                if ekfilter.first == False:
                    state[1] = 0
                    state[3] = 0
                    state[5] = 0

                state[0] = armor.in_imu[0]
                state[2] = armor.in_imu[1]
                state[4] = armor.in_imu[2]

                ptsEKF = ekfilter.step(deltaTime, (robot.yaw, robot.pitch), state, armor.observation)
                predictedPtsInWorld = ekfilter.getCompensatedPtsInWorld(ptsEKF, 50, robot.bullet_speed, 3)  # predictTime后目标在世界坐标系下的坐标(mm)

                cx, cy, cz = armor.in_camera.T[0]
                x, y, z = armor.in_imu.T[0]
                px, py, pz = ptsEKF.T[0]
                ppx, ppy, ppz = predictedPtsInWorld
                _state = ekfilter.state.T[0]
                vx, vy, vz = _state[1], _state[3], _state[5]

                # robot.send(x, y, z)
                # robot.send(px, py, pz)
                robot.send(ppx, ppy-60, ppz)
                

                # 调试用
                # visualizer.plot((cy, y, robot.yaw*10, robot.pitch*10), ('cy', 'y', 'yaw', 'pitch'))
                visualizer.plot((x, y, z, robot.yaw*10, robot.pitch*10), ('x', 'y', 'z', 'yaw', 'pitch'))
                # visualizer.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz'))
                # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz,vx*10,vy*10,vz*10), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz','vx','vy','vz'))
