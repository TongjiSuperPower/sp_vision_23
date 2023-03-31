import cv2
import time
import numpy as np
from collections import deque

import modules.tools as tools
from modules.robot import Robot
from modules.ExtendKF import EKF
from modules.armor_detection import ArmorDetector

from remote_visualizer import Visualizer
from configs.infantry3 import cameraMatrix, distCoeffs, cameraVector

if __name__ == '__main__':
    robot = Robot(3, '/dev/ttyUSB0')
    visualizer = Visualizer()

    armor_detector = ArmorDetector(cameraMatrix, distCoeffs, cameraVector)

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

        # 调试用
        drawing = img.copy()
        for l in armor_detector._filtered_lightbars:
            tools.drawContour(drawing, l.points, (255, 255, 0), 5)
        for a in armors:
            cx, cy, cz = a.in_camera.T[0]
            tools.drawContour(drawing, a.points)
            tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
            tools.putText(drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))
        visualizer.show(drawing)

        if len(armors) == 0:
            lostFrame += 1
            if lostFrame > maxLostFrame:
                # after losing armor for a while
                lostFrame = 0
                ekfilter = EKF(6, 3)  # create a new filter
                print(f'Lost Over {maxLostFrame} Frames! {continuousFrameCount=}')
                continuousFrameCount = 0

        if len(armors) > 0:
            continuousFrameCount += 1

            twoTimeStampMs.append(robot.camera_stamp_ms)
            deltaTime = (twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 5

            armor = armors[0]
            predictedPtsInWorld = armor.in_imu.T[0]  # 如果没开EKF，就发送识别值

            twoPtsInCam.append(armor.in_camera)
            twoPtsInTripod.append(armor.in_gimbal)
            twoPtsInWorld.append(armor.in_imu)  # mm

            rvx, rvy, rvz = 0, 0, 0

            if ekfilter.first == False:
                # state[1] = (twoPtsInWorld[1][0] - twoPtsInWorld[0][0])/deltaTime # m/s
                # state[3] = (twoPtsInWorld[1][1] - twoPtsInWorld[0][1])/deltaTime
                # state[5] = (twoPtsInWorld[1][2] - twoPtsInWorld[0][2])/deltaTime
                state[1] = 0
                state[3] = 0
                state[5] = 0
                rvx, rvy, rvz = state[1][0], state[3][0], state[5][0]

            state[0] = armor.in_imu[0]
            state[2] = armor.in_imu[1]
            state[4] = armor.in_imu[2]

            ptsEKF = ekfilter.step(deltaTime, (robot.yaw, robot.pitch), state, armor.observation)
            predictedPtsInWorld = ekfilter.getCompensatedPtsInWorld(ptsEKF, 20, 10, 0)  # predictTime后目标在世界坐标系下的坐标(mm)
            robot.send(*predictedPtsInWorld)

            # 调试用
            # cx, cy, cz = armor.in_camera.T[0]
            # x, y, z = armor.in_imu.T[0]
            # px, py, pz = ptsEKF.T[0]
            # _state = ekfilter.state.T[0]
            # vx, vy, vz = _state[1], _state[3], _state[5]
            # v.plot((cx, x, r.yaw*10), ('cx', 'x', 'yaw'))
            # v.plot((r.yaw, r.pitch), ('yaw', 'pitch'))
            # v.plot((x, y, z, r.yaw*10, r.pitch*10), ('x', 'y', 'z', 'yaw', 'pitch'))
            # v.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
            # v.plot((x, y, z), ('x', 'y', 'z'))
            # v.plot((vx,vy,vz), ('x', 'y', 'z'))
