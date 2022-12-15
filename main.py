import cv2
import time
import numpy as np

from modules.mindvision import Camera
from modules.detection import Detector
from modules.communication import Communicator
from modules.utilities import drawContour, drawPoint, drawAxis, putText

# TODO config.toml
debug = True
useCamera = False
useSerial = False
port = '/dev/tty.usbserial-A50285BI'  # for ubuntu: '/dev/ttyUSB0'
cameraMatrix = np.float32([[1.30161072e+03, 0, 6.65920641e+02],
                           [0, 1.30289452e+03, 5.09983987e+02],
                           [0, 0, 1]])
distCoeffs = np.float32([-4.98089049e-01, 4.27962976e-01, -3.56114307e-03, -3.35744316e-04, -5.79237391e-01])

# TODO 大装甲板
lightBarLength, armorWidth = 56, 135
objPoints = np.float32([[-armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, -lightBarLength / 2, 0],
                        [armorWidth / 2, lightBarLength / 2, 0],
                        [-armorWidth / 2, lightBarLength / 2, 0]])

cap = Camera() if useCamera else cv2.VideoCapture('assets/input.avi')
detector = Detector()
if useSerial:
    communicator = Communicator(port)
if debug:
    output = cv2.VideoWriter('assets/output.avi', cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'), 30, (1280, 1024))

while True:
    success, frame = cap.read()
    if not success:
        break

    start = time.time()

    lightBars, armors = detector.detect(frame)

    if len(armors) > 0:
        a = armors[0]  # TODO a = classifior.classify(armors)

        a.targeted(objPoints, cameraMatrix, distCoeffs)

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
