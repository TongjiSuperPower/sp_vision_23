import cv2
import time
import numpy as np
import os
import sys
import toml

from modules.mindvision import Camera
from modules.detection import Detector
from modules.communication import Communicator
from modules.utilities import drawContour, drawPoint, drawAxis, putText

def readConfig():
    '''读取配置文件'''    
    cfgFile = 'assets/camConfig.toml'

    if not os.path.exists(cfgFile):
        print(cfgFile + ' not found')
        sys.exit(-1)
    
    content = toml.load(cfgFile) 

    cameraMatrix = np.float32(content['cameraMatrix'])  
    distCoeffs = np.float32(content['distCoeffs']) 

    return [cameraMatrix, distCoeffs]

# TODO config.toml
debug = True
useCamera = True
exposureMs = 0.5
useSerial = False
enablePredict = False # 开启KF滤波与预测
savePts = True # 是否把相机坐标系下的坐标保存txt文件
port = '/dev/tty.usbserial-A50285BI'  # for ubuntu: '/dev/ttyUSB0'

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
    
while True:
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

        # TODO yaw, pitch = predictor.predict(a)

        if debug:
            drawAxis(frame, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            putText(frame, f'{a.yaw:.2f} {a.pitch:.2f}', a.center)
            drawPoint(frame, a.center, (255, 255, 255))

        if useSerial:
            communicator.send(a.yaw, -a.pitch * 0.5)
    else:
        if useSerial:
            communicator.send(0, 0)

    processTimeMs = (time.time() - start) * 1000
    print(f'{processTimeMs=}')

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

cap.release()

if debug:
    output.release()
if savePts:
    txtFile.close()

