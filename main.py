import matplotlib as mpl
mpl.use('TkAgg')
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
import enum

tVecFromCam2Tri = np.array([0, 60, 50])

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
    ptsInTripod = ptsInCam + tVecFromCam2Tri
    return ptsInTripod


def ptsInTripod2World(ptsInTripod:np.ndarray, yaw:float, pitch:float):
    '''云台坐标系->世界坐标系。yaw、pitch为角度'''
    yaw = yaw/180.0*math.pi
    pitch = pitch/180.0*math.pi
    yRotationMatrix = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    xRotationMatrix = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ptsInWorld = np.dot(yRotationMatrix, np.dot(xRotationMatrix, np.reshape(ptsInTripod,(3,1)))).T
    return np.reshape(ptsInWorld,(3,))


def ptsInCam2World(ptsInCam, yaw, pitch):
    '''相机坐标系->世界坐标系。yaw、pitch为角度'''
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

def getInvMatrix(matrix: np.ndarray)->np.ndarray:   
    u, s, v = np.linalg.svd(matrix, full_matrices=False)
    inv = np.matmul(v.T * 1 / s, u.T)
    return inv

def ptsInWorld2Img(ptsInWorld:np.ndarray, yaw:float, pitch:float, rvec, cameraMatrix, distCoeffs)->np.ndarray:
    '''世界坐标系->图像坐标系。yaw、pitch为角度'''
    # 世界坐标系->云台坐标系：
    yaw = yaw/180.0*math.pi
    pitch = pitch/180.0*math.pi
    yRotationMatrix = np.array([[math.cos(yaw), 0, math.sin(yaw)], [0, 1, 0], [-math.sin(yaw), 0, math.cos(yaw)]])
    xRotationMatrix = np.array([[1, 0, 0], [0, math.cos(pitch), -math.sin(pitch)], [0, math.sin(pitch), math.cos(pitch)]])
    ptsInTripod = np.dot(getInvMatrix(xRotationMatrix), np.dot(getInvMatrix(yRotationMatrix),np.reshape(ptsInWorld,(3,1)))).T
    # 云台坐标系->相机坐标系：
    ptsInCam = np.reshape(ptsInTripod,(3,)) - tVecFromCam2Tri 
    # 相机坐标系->图像坐标系：
    tvec = ptsInCam
    ptsInImg,_ = cv2.projectPoints(ptsInCam, rvec, tvec, cameraMatrix, distCoeffs)
    return np.reshape(ptsInImg,(2,))


#################################################
debug = False
useCamera = True
exposureMs = 1 # 相机曝光时间(ms)
useSerial = True
enablePredict = False  # 开启KF滤波与预测
savePts = True # 是否把相机坐标系下的坐标保存txt文件
enableDrawKF = False
functionType = FunctionType.autoaim
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
class VisualComu():
    yaw=10.
    pitch=2.
    def __init__(self) -> None:
        pass
detector = Detector()
if useSerial:
    communicator = Communicator(port)
else:
    communicator = VisualComu()
if debug:
    output = cv2.VideoWriter('assets/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 1024))
if savePts:
    txtFile = open('assets/ptsInCam.txt', mode='w')
    timeFile = open('assets/time.txt', mode='w')
    totalTime = []
if enablePredict:
    ekfilter = EKF(6, 3)
    maxLostFrame = 3  # 最大丢失帧数
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
    drawDx,drawDy,drawDz=[],[],[]
    drawEKFx, drawEKFy, drawEKFz = [],[],[]
    plt.ion()
    # aFig = plt.subplot(2, 1, 1)
    # bFig = plt.subplot(2, 1, 2)

while True:
    if not useSerial or communicator.received():
        success, frame = cap.read()
        if not success:
            break

        timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() * 1e6)
        twoTimeStampUs.append(timeStampUs)

        deltaTime = (twoTimeStampUs[1] - twoTimeStampUs[0])/1e3 if len(twoTimeStampUs) == 2 else 5  # ms

        totalTime.append(deltaTime)

        if functionType == FunctionType.autoaim :

            lightBars, armors = detector.detect(frame)

            if len(armors) > 0:
                a = armors[0]  # TODO a = classifior.classify(armors)

                a.targeted(objPoints, cameraMatrix, distCoeffs)
                predictedPtsInWorld = ptsInCam2World(np.reshape(a.aimPoint,(3,)),communicator.yaw,communicator.pitch)

                if savePts:
                    x, y, z = a.aimPoint
                    txtFile.write(str(x) + " " + str(y) + " " + str(z) + " \n")
                    timeFile.write(str(deltaTime)+"\n")


                if not useSerial:
                    communicator.yaw = 0
                    communicator.pitch = 0

                if enablePredict:
                    x, y, z = a.aimPoint
                    ptsInCam = [x, y, z]
                    ptsInTripod = ptsInCam2Tripod(ptsInCam)
                    ptsInWorld = ptsInTripod2World(ptsInTripod, communicator.yaw, communicator.pitch)
                    observation = getObservation(ptsInCam)

                    twoPtsInCam.append(ptsInCam)
                    twoPtsInTripod.append(ptsInTripod)
                    twoPtsInWorld.append(ptsInWorld) # mm

                    
                    print("time:")
                    print(deltaTime)
                    print("\n")

                    if ekfilter.first == False:
                        state[1] = (twoPtsInWorld[1][0] - twoPtsInWorld[0][0])/deltaTime # m/s
                        state[3] = (twoPtsInWorld[1][1] - twoPtsInWorld[0][1])/deltaTime
                        state[5] = (twoPtsInWorld[1][2] - twoPtsInWorld[0][2])/deltaTime

                    state[0] = ptsInWorld[0]
                    state[2] = ptsInWorld[1]
                    state[4] = ptsInWorld[2]

                    ptsEKF = ekfilter.step(deltaTime, [communicator.yaw, communicator.pitch], state, observation, np.reshape(ptsInCam, (3, 1)))
                    ptsEKF = ptsEKF.T
                    

                    predictTime = 10  # ms
                    bulletSpeed = 15  # TODO 测延迟和子弹速度
                    predictedYaw, predictedPitch = ekfilter.predict(predictTime, bulletSpeed)                    
                    
                    predictedPtsInWorld = ekfilter.getCompensatedPtsInWorld(ptsEKF, 10, 15) # predictTime后目标在世界坐标系下的坐标(mm)
                    dx = ekfilter.state[1]
                    dy = ekfilter.state[3]
                    dz = ekfilter.state[5]


                    if enableDrawKF:
                        drawXAxis.append(drawCount)
                        drawCount += 1
                        drawYaw.append(a.yaw)
                        drawPredictedYaw.append(predictedYaw)
                        drawPitch.append(a.pitch)
                        drawPredictedPitch.append(predictedPitch)

                        plt.clf()

                        # plt.plot(drawXAxis, drawYaw, label='yaw')
                        # plt.plot(drawXAxis, drawPredictedYaw, label='Pyaw')
                        # plt.plot(drawXAxis, drawPitch, label='pitch')
                        # plt.plot(drawXAxis, drawPredictedPitch, label='Ppitch')
                        # plt.legend()

                  

                        aPtsInWorld = ptsInCam2World(a.aimPoint, communicator.yaw, communicator.pitch)

                        drawX.append(aPtsInWorld[0])
                        drawY.append(aPtsInWorld[1])
                        drawZ.append(aPtsInWorld[2])

                    
                        drawPredictedX.append(predictedPtsInWorld[0])
                        drawPredictedY.append(predictedPtsInWorld[1])
                        drawPredictedZ.append(predictedPtsInWorld[2])
                     

                        # plt.plot(drawXAxis, drawX, label='x')
                        # plt.plot(drawXAxis, drawY, label='y')
                        # plt.plot(drawXAxis, drawZ, label='z')
                        # plt.plot(drawXAxis, drawPredictedX, label='Px')
                        # plt.plot(drawXAxis, drawPredictedY, label='Py')
                        # plt.plot(drawXAxis, drawPredictedZ, label='Pz')

                        drawEKF = np.reshape(ptsEKF, (3,))
                        drawEKFx.append(drawEKF[0])
                        drawEKFy.append(drawEKF[1])
                        drawEKFz.append(drawEKF[2])
                        # plt.plot(drawXAxis, drawEKFx, label='Ex')
                        # plt.plot(drawXAxis, drawEKFy, label='Ey')
                        # plt.plot(drawXAxis, drawEKFz, label='Ez')
                      

                        drawDx.append(dx)
                        drawDy.append(dy)
                        drawDz.append(dz)

                        plt.plot(drawXAxis, drawDx, label='dx')
                        plt.plot(drawXAxis, drawDy, label='dy')
                        plt.plot(drawXAxis, drawDz, label='dz')

                        plt.legend()                      

                        plt.pause(0.001)

                if debug:
                    drawAxis(frame, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
                    putText(frame, f'{a.yaw:.2f} {a.pitch:.2f}', a.center)
                    drawPoint(frame, a.center, (0, 255, 0))                    
                    prePtsInImg = ptsInWorld2Img(predictedPtsInWorld, communicator.yaw, communicator.pitch, a.rvec, cameraMatrix, distCoeffs)                    
                           
                    ptsInImg,_ = cv2.projectPoints(a.aimPoint, a.rvec, ptsInCam2World(a.aimPoint,communicator.yaw,communicator.pitch), cameraMatrix, distCoeffs)
                    drawPoint(frame,np.reshape(ptsInImg,(2,)), (0,0,255))
                    drawPoint(frame,np.reshape(prePtsInImg,(2,)), (255,0,0))

                if useSerial:
                    if enablePredict:
                        communicator.send(*predictedPtsInWorld)
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
    timeFile.close()
if enableDrawKF:
    plt.ioff()
