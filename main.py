from modules.ExtendKF import EKF
from modules.utilities import drawContour, drawPoint, drawAxis, putText
from modules.communication import Communicator
from modules.detection import Detector
from modules.mindvision import Camera
from modules.Nahsor.Nahsor import *
from collections import deque
import math
import toml
import sys
import os
import numpy as np
import time
import cv2
from matplotlib import pyplot as plt
import matplotlib as mpl
mpl.use('TkAgg')
import enum

class FunctionType(enum.Enum):
    '''系统工作模式'''
    autoaim = 1
    smallEnergy = 2
    bigEnergy = 3

def readConfig():
    '''读取配置文件'''
    cfgFile = 'assets/camConfig2.toml'

    if not os.path.exists(cfgFile):
        print(cfgFile + ' not found')
        sys.exit(-1)

    content = toml.load(cfgFile)

    cameraMatrix = np.float32(content['cameraMatrix'])
    distCoeffs = np.float32(content['distCoeffs'])

    return [cameraMatrix, distCoeffs]


def ptsInCam2Tripod(ptsInCam):
    '''相机坐标系->云台坐标系'''
    ptsInTripod = ptsInCam + np.array([0, 60, 50])
    return ptsInTripod


def ptsInTripod2World(ptsInTripod, yaw, pitch):
    yRotationMatrix = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    xRotationMatrix = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ptsInWorld = np.dot(yRotationMatrix, np.dot(xRotationMatrix, ptsInTripod)).T
    return ptsInWorld


def ptsInCam2World(ptsInCam, yaw, pitch):
    ptsInTripod = ptsInCam2Tripod(ptsInCam)
    ptsInWorld = ptsInTripod2World(ptsInTripod, yaw, pitch)
    return ptsInWorld


def getObservation(ptsInCam):
    x = ptsInCam[0]
    y = ptsInCam[1]
    z = ptsInCam[2]
    alpha = math.atan(x/z)
    beta = math.atan(y/z)
    observation = [z, alpha, beta]
    return observation

#################################################
debug = True
useCamera = True
exposureMs = 1 # 相机曝光时间
useSerial = False
enablePredict = False  # 开启KF滤波与预测
savePts = False  # 是否把相机坐标系下的坐标保存txt文件
enableDrawKF = False
functionType = FunctionType.smallEnergy
port = '/dev/ttyUSB0'  # for ubuntu: '/dev/ttyUSB0'
#################################################

[cameraMatrix, distCoeffs] = readConfig()

# TODO 大装甲板
lightBarLength, armorWidth = 56, 135
objPoints = np.float32([[-armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, lightBarLength / 2, 0],
                        [-armorWidth / 2, lightBarLength / 2, 0]])

cap = Camera(exposureMs) if useCamera else cv2.VideoCapture('assets/input.avi')
detector = Detector()
if useSerial:
    communicator = Communicator(port)
if debug:
    output = cv2.VideoWriter('assets/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 1024))
if savePts:
    txtFile = open('assets/ptsInCam.txt', mode='w')
if enablePredict:
    ekfilter = EKF(6, 3)
    maxLostFrame = 4  # 最大丢失帧数
    lostFrame = 0  # 丢失帧数
    state = np.zeros((6, 1))
    twoPtsInCam = deque(maxlen=2)  # a queue with max 2 capaticity
    twoPtsInWorld = deque(maxlen=2)
    twoPtsInTripod = deque(maxlen=2)
    twoTimeStampUs = deque(maxlen=2)
if enableDrawKF:
    drawCount = 0
    drawXAxis, drawYaw, drawPredictedYaw, drawPitch, drawPredictedPitch = [], [], [], [], []
    drawX, drawPredictedX, drawY, drawPredictedY, drawZ, drawPredictedZ = [], [], [], [], [], []
    plt.ion()
    aFig = plt.subplot(2, 1, 1)
    bFig = plt.subplot(2, 1, 2)

while True:
    if not useSerial or communicator.received():
        success, frame = cap.read()
        if not success:
            break

        if functionType == FunctionType.autoaim :

            lightBars, armors = detector.detect(frame)

            if len(armors) > 0:
                a = armors[0]  # TODO a = classifior.classify(armors)

                a.targeted(objPoints, cameraMatrix, distCoeffs)

                if savePts:
                    x, y, z = a.aimPoint
                    txtFile.write(str(x) + " " + str(y) + " " + str(z) + " \n")

                if enablePredict:
                    ptsInCam = [x, y, z]
                    ptsInTripod = ptsInCam2Tripod(ptsInCam)
                    ptsInWorld = ptsInTripod2World(ptsInTripod, communicator.yaw, communicator.pitch)
                    observation = getObservation(ptsInCam)

                    twoPtsInCam.append(ptsInCam)
                    twoPtsInTripod.append(ptsInTripod)
                    twoPtsInWorld.append(ptsInWorld)

                    timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() * 1e6)
                    twoTimeStampUs.append(timeStampUs)

                    deltaTime = (twoTimeStampUs[1] - twoTimeStampUs[0])*1e3 if len(twoTimeStampUs) == 2 else 10  # ms

                    if ekfilter.first == False:
                        state[1] = (twoPtsInWorld[1][0] - twoPtsInWorld[0][0])/deltaTime
                        state[3] = (twoPtsInWorld[1][1] - twoPtsInWorld[0][1])/deltaTime
                        state[5] = (twoPtsInWorld[1][2] - twoPtsInWorld[0][2])/deltaTime

                    state[0] = ptsInWorld[0]
                    state[2] = ptsInWorld[1]
                    state[4] = ptsInWorld[2]

                    predictedPtsInWorld = ekfilter.step(deltaTime, [communicator.yaw, communicator.pitch], state, observation, np.reshape(ptsInCam, (3, 1)))
                    ptsEKF = predictedPtsInWorld.T
                    print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                    print(ptsEKF)

                    predictTime = 2000*1e-3  # ms
                    bulletSpeed = 5  # TODO 测延迟和子弹速度
                    predictedYaw, predictedPitch = ekfilter.predict(predictTime, bulletSpeed)

                    if enableDrawKF:
                        drawXAxis.append(drawCount)
                        drawCount += 1
                        drawYaw.append(a.yaw)
                        drawPredictedYaw.append(predictedYaw)
                        drawPitch.append(a.pitch)
                        drawPredictedPitch.append(predictedPitch)

                        plt.clf()

                        plt.plot(drawXAxis, drawYaw, label='yaw')
                        plt.plot(drawXAxis, drawPredictedYaw, label='Pyaw')
                        plt.plot(drawXAxis, drawPitch, label='pitch')
                        plt.plot(drawXAxis, drawPredictedPitch, label='Ppitch')
                        plt.legend()

                        print('1\n')

                        aPtsInWorld = ptsInCam2World(a.aimPoint, a.yaw, a.pitch)

                        drawX.append(aPtsInWorld[0])
                        drawY.append(aPtsInWorld[1])
                        drawZ.append(aPtsInWorld[2])

                        print('2\n')

                        drawPredictedX.append(ptsEKF[0][0])
                        drawPredictedY.append(ptsEKF[0][1])
                        drawPredictedZ.append(ptsEKF[0][2])

                        print('3\n')

                        bFig.plot(drawXAxis, drawX, label='x')
                        bFig.plot(drawXAxis, drawY, label='y')
                        bFig.plot(drawXAxis, drawZ, label='z')
                        bFig.plot(drawXAxis, drawPredictedX, label='Px')
                        bFig.plot(drawXAxis, drawPredictedY, label='Py')
                        bFig.plot(drawXAxis, drawPredictedZ, label='Pz')
                        bFig.legend()

                        print('4\n')

                        plt.pause(0.001)

                if debug:
                    drawAxis(frame, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                    putText(frame, f'{a.yaw:.2f} {a.pitch:.2f}', a.center)
                    drawPoint(frame, a.center, (255, 255, 255))

                if useSerial:
                    if enablePredict:
                        communicator.send_yaw_pitch(communicator.yaw - predictedYaw, communicator.pitch - predictedPitch)  # 这里未验证方向是否正确
                    else:
                        target_in_gimabl = np.array([a.aimPoint]).T
                        yaw, pitch = communicator.yaw / 180 * math.pi, communicator.pitch / 180 * math.pi
                        yRotationMatrix = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                                                    [0, 1, 0],
                                                    [-math.sin(yaw), 0, math.cos(yaw)]])
                        xRotationMatrix = np.array([[1, 0, 0],
                                                    [0, math.cos(pitch), -math.sin(pitch)],
                                                    [0, math.sin(pitch), math.cos(pitch)]])
                        target_in_world = (yRotationMatrix @ xRotationMatrix @ target_in_gimabl).T[0]
                        communicator.send(*target_in_world)

            else:
                if enablePredict:
                    # TODO 通过数字识别判断装甲板ID号，从而制定filter重置逻辑
                    lostFrame += 1
                    if lostFrame > maxLostFrame:
                        # after losing armor for a while
                        lostFrame = 0
                        ekfilter = EKF(6, 3)  # create a new filter

            if debug:
                for l in lightBars:
                    drawContour(frame, l.points, (0, 255, 255), 10)
                for a in armors:
                    drawContour(frame, a.points)
                cv2.imshow('result', frame)
                output.write(frame)

                if (cv2.waitKey(30) & 0xFF) == ord('q'):
                    break

            if useSerial:
                communicator.reset_input_buffer()

        if functionType == FunctionType.smallEnergy:
            # 新建能量机关对象
            color = 'R'
            w = NahsorMarker(color, 20, debug=1, get_R_method=0)   # 传入参数为 1.颜色代码：B/b -> 蓝色,  R/r -> 红色;

            # 帧率计算
            nowt = time.time()
            last = time.time()
            rfps = 0
            pfps = 0
            while (cv2.waitKey(1) & 0xFF) != ord('q'):
                rfps = rfps + 1
                nowt = time.time()
                if nowt - last >= 1:
                    last = nowt
                    # print(rfps)
                    pfps = rfps
                    rfps = 0

                img = frame

                # 显示帧率
                img = cv2.putText(img, str(pfps), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

                # 使用mark()方法，传入一帧图像
                w.mark(img)

                # 使用markFrame()获得标记好的输出图像
                img = w.markFrame()

                # 使用getResult()方法获得输出
                # print(w.getResult())

                cv2.imshow("Press q to end", img)
           
cap.release()
if debug:
    output.release()
if savePts:
    txtFile.close()
if enableDrawKF:
    plt.ioff()
