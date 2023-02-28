import matplotlib as mpl
mpl.use('TkAgg')
from matplotlib import pyplot as plt

import cv2
import time
import numpy as np
import os
import sys
import toml
import math
from collections import deque

from modules.mindvision import Camera
from modules.detection import Detector
from modules.communication import Communicator
from modules.utilities import drawContour, drawPoint, drawAxis, putText
from modules.ExtendKF import EKF

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
    yRotationMatrix = np.array([[math.cos(yaw),0,math.sin(yaw)],[0,1,0],[-math.sin(yaw),0,math.cos(yaw)]])
    xRotationMatrix = np.array([[1,0,0],[0,math.cos(pitch),-math.sin(pitch)],[0,math.sin(pitch),math.cos(pitch)]])
    ptsInWorld = np.dot(yRotationMatrix, np.dot(xRotationMatrix,ptsInTripod)).T
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

debug = True
useCamera = True
exposureMs = 0.5
useSerial = True
enablePredict = True # 开启KF滤波与预测
savePts = True # 是否把相机坐标系下的坐标保存txt文件
enableDrawKF = True
port = '/dev/ttyUSB0'  # for ubuntu: '/dev/ttyUSB0'

[cameraMatrix,distCoeffs] = readConfig()

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
    ekfilter = EKF(6,3)
    maxLostFrame = 4 # 最大丢失帧数
    lostFrame = 0 # 丢失帧数
    state = np.zeros((6,1))
    twoPtsInCam = deque(maxlen=2) # a queue with max 2 capaticity
    twoPtsInWorld = deque(maxlen=2)
    twoPtsInTripod = deque(maxlen=2)
    twoTimeStampUs = deque(maxlen=2)
if enableDrawKF:
    drawCount = 0
    drawXAxis, drawYaw, drawPredictedYaw, drawPitch, drawPredictedPitch = [],[],[],[],[]
    drawX, drawPredictedX, drawY, drawPredictedY, drawZ, drawPredictedZ= [],[],[],[],[],[]
    plt.ion()
    # aFig = plt.subplot(2,1,1)
    # bFig = plt.subplot(2,1,2)

while True:
    if communicator.received():

        success, frame = cap.read()
        if not success:
            break

        start = time.time()

        lightBars, armors = detector.detect(frame)

        if len(armors) > 0:
            a = armors[0]  # TODO a = classifior.classify(armors)

            a.targeted(objPoints, cameraMatrix, distCoeffs)
            
            if savePts:
                x,y,z = a.aimPoint            
                txtFile.write(str(x) +" "+ str(y) +" "+ str(z) +" \n")

            if enablePredict:
                ptsInCam = [x,y,z]
                ptsInTripod = ptsInCam2Tripod(ptsInCam)
                ptsInWorld = ptsInTripod2World(ptsInTripod, communicator.yaw, communicator.pitch)
                observation = getObservation(ptsInCam)

                twoPtsInCam.append(ptsInCam)
                twoPtsInTripod.append(ptsInTripod)
                twoPtsInWorld.append(ptsInWorld)

                timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() *1e6)
                twoTimeStampUs.append(timeStampUs)

                deltaTime = (twoTimeStampUs[1] - twoTimeStampUs[0])*1e3 if len(twoTimeStampUs)==2 else 10 # ms

                if ekfilter.first==False:
                    state[1] = (twoPtsInWorld[1][0] - twoPtsInWorld[0][0])/deltaTime
                    state[3] = (twoPtsInWorld[1][1] - twoPtsInWorld[0][1])/deltaTime
                    state[5] = (twoPtsInWorld[1][2] - twoPtsInWorld[0][2])/deltaTime
                
                state[0] = ptsInWorld[0]
                state[2] = ptsInWorld[1]
                state[4] = ptsInWorld[2]

                predictedPtsInWorld = ekfilter.step(deltaTime, [communicator.yaw,communicator.pitch], state, observation, np.reshape(ptsInCam, (3,1)))
                ptsEKF = predictedPtsInWorld.T
                print('!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!111')
                print(ptsEKF)

                predictTime = 2000*1e-3 #ms
                bulletSpeed = 5 # TODO 测延迟和子弹速度
                predictedYaw, predictedPitch = ekfilter.predict(predictTime, bulletSpeed)     

                if enableDrawKF:
                    # drawXAxis.append(drawCount)
                    # drawCount += 1
                    # drawYaw.append(a.yaw)
                    # drawPredictedYaw.append(predictedYaw)
                    # drawPitch.append(a.pitch)
                    # drawPredictedPitch.append(predictedPitch)

                    plt.clf()
                    
                    # plt.plot(drawXAxis,drawYaw,label='yaw')
                    # plt.plot(drawXAxis, drawPredictedYaw,label='Pyaw')
                    # plt.plot(drawXAxis,drawPitch,label='pitch')
                    # plt.plot(drawXAxis, drawPredictedPitch,label='Ppitch')
                    # plt.legend()

                    # print('1\n')

                    # aPtsInWorld = ptsInCam2World(a.aimPoint,a.yaw,a.pitch)

                    # drawX.append(aPtsInWorld[0])
                    # drawY.append(aPtsInWorld[1])
                    # drawZ.append(aPtsInWorld[2])

                    # print('2\n')

                    # drawPredictedX.append(ptsEKF[0][0])
                    # drawPredictedY.append(ptsEKF[0][1])
                    # drawPredictedZ.append(ptsEKF[0][2])

                    # print('3\n')

                    # # bFig.plot(drawXAxis, drawX, label='x')
                    # # bFig.plot(drawXAxis, drawY, label='y')
                    # # bFig.plot(drawXAxis, drawZ, label='z')
                    # # bFig.plot(drawXAxis, drawPredictedX, label='Px')
                    # # bFig.plot(drawXAxis, drawPredictedY, label='Py')
                    # # bFig.plot(drawXAxis, drawPredictedZ, label='Pz')
                    # # bFig.legend()



                    plt.plot([1,2,3])
                    # # plt.plot(drawXAxis, drawY, label='y')
                    # # plt.plot(drawXAxis, drawZ, label='z')
                    # # plt.plot(drawXAxis, drawPredictedX, label='Px')
                    # # plt.plot(drawXAxis, drawPredictedY, label='Py')
                    # # plt.plot(drawXAxis, drawPredictedZ, label='Pz')
                   
                    # print('4\n')

                    plt.pause(0.001)

                         

            if debug:
                drawAxis(frame, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                putText(frame, f'{a.yaw:.2f} {a.pitch:.2f}', a.center)
                drawPoint(frame, a.center, (255, 255, 255))

            if useSerial:
                if enablePredict:
                    communicator.send(communicator.yaw - predictedYaw, communicator.pitch - predictedPitch)
                else:
                    communicator.send(communicator.yaw - a.yaw, communicator.pitch - a.pitch)
        else:
            if useSerial:
                communicator.send(communicator.yaw, communicator.pitch)
            
            if enablePredict:
                # TODO 通过数字识别判断装甲板ID号，从而制定filter重置逻辑
                lostFrame += 1
                if lostFrame > maxLostFrame:
                    # after losing armor for a while
                    lostFrame = 0
                    ekfilter = EKF(6,3) # create a new filter


        processTimeMs = (time.time() - start) * 1000
        # print(f'{processTimeMs=}')

        if debug:
            for l in lightBars:
                drawContour(frame, l.points, (0, 255, 255), 10)
            for a in armors:
                drawContour(frame, a.points)
            cv2.convertScaleAbs(frame, frame, alpha=5)
            cv2.imshow('result', frame)
            output.write(frame)

            if (cv2.waitKey(30) & 0xFF) == ord('q'):
                break

        communicator.reset_input_buffer()
    else:
        continue

cap.release()
if debug:
    output.release()
if savePts:
    txtFile.close()
if enableDrawKF():
    plt.ioff()

