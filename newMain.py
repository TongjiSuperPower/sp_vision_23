import cv2
import time
import numpy as np

import modules.tools as tools
import modules.robotSystem as robotSystem
from modules.classification import Classifier
from modules.armor_detection import ArmorDetector
from modules.communication import Communicator
from collections import deque
from modules.mindvision import Camera
from modules.ExtendKF import EKF

#################################################
debug = True
useCamera = True
exposureMs = 4 # 相机曝光时间(ms)
useSerial = False
enablePredict = False # 开启KF滤波与预测
savePts = False # 是否把相机坐标系下的坐标保存txt文件
enableDrawKF = False
functionType = robotSystem.FunctionType.autoaim
port = '/dev/ttyUSB0'  # for ubuntu: '/dev/ttyUSB0'
#################################################


if __name__ == '__main__':
    from configs.infantry3 import cameraMatrix, distCoeffs, cameraVector

    robot = robotSystem.Robot()

    video_path = 'assets/input.avi'
    cap = Camera(exposureMs) if useCamera else cv2.VideoCapture(video_path)
    classifier = Classifier()
    armor_detector = ArmorDetector(cameraMatrix, distCoeffs, cameraVector, classifier)

    if useSerial:
        communicator = Communicator(port)
    else:
        communicator = tools.VirtualComu()

    if enablePredict:
        ekfilter = EKF(6, 3)
        maxLostFrame = 3  # 最大丢失帧数
        lostFrame = 0  # 丢失帧数
        state = np.zeros((6, 1))
        twoPtsInCam = deque(maxlen=2)  # a queue with max 2 capaticity
        twoPtsInWorld = deque(maxlen=2)
        twoPtsInTripod = deque(maxlen=2)

    twoTimeStampUs = deque(maxlen=2)
    totalTime = []

    while True:
        if useSerial and not communicator.received():
            break
        success, frame = cap.read()
        if not success:
            break

        timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() * 1e6)
        twoTimeStampUs.append(timeStampUs)
        deltaTime = (twoTimeStampUs[1] - twoTimeStampUs[0])/1e3 if len(twoTimeStampUs) == 2 else 5  # ms
        totalTime.append(deltaTime)

        if functionType == robotSystem.FunctionType.autoaim:
            armors = armor_detector.detect(frame, 0, 0)

            if armors.count == 0 and enablePredict:
                lostFrame += 1
                if lostFrame > maxLostFrame:
                    # after losing armor for a while
                    lostFrame = 0
                    ekfilter = EKF(6, 3)  # create a new filter


            drawing = frame.copy()
            for a in armors:
                tools.drawContour(drawing, a.points)
                tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                x, y, z = a.in_imu.T[0]
                tools.putText(drawing, f'x{x:.1f} y{y:.1f} z{z:.1f}', a.left.bottom, (255, 255, 255))

            cv2.imshow('press q to exit', drawing)

            # 显示所有图案图片
            for i, a in enumerate(armor_detector._armors):
                cv2.imshow(f'{i}', a.pattern)

            armor = armors[0]
            predictedPtsInWorld = armor.in_imu # 如果没开EKF，就发送识别值

            if enablePredict:
                twoPtsInCam.append(armor.in_camera)
                twoPtsInTripod.append(armor.in_gimbal)
                twoPtsInWorld.append(armor.in_imu) # mm
                
                if ekfilter.first == False:
                    state[1] = (twoPtsInWorld[1][0] - twoPtsInWorld[0][0])/deltaTime # m/s
                    state[3] = (twoPtsInWorld[1][1] - twoPtsInWorld[0][1])/deltaTime
                    state[5] = (twoPtsInWorld[1][2] - twoPtsInWorld[0][2])/deltaTime

                state[0] = armor.in_imu[0]
                state[2] = armor.in_imu[1]
                state[4] = armor.in_imu[2]

                ptsEKF = ekfilter.step(deltaTime, [communicator.yaw, communicator.pitch], state, armor.observation)
                predictedPtsInWorld = ekfilter.getCompensatedPtsInWorld(ptsEKF, robot.delayTime, robot.bulletSpeed, 0) # predictTime后目标在世界坐标系下的坐标(mm)
                    

            if useSerial:
                communicator.send(*predictedPtsInWorld)

            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break

    cap.release()
