# == 测试反小陀螺的demo == #

import cv2
import time
import numpy as np

import modules.tools as tools
from modules.armor_detection import ArmorDetector
from modules.antitop import TopStateDeque

if __name__ == '__main__':
    from configs.infantry3 import cameraMatrix, distCoeffs, cameraVector

    video_path = 'assets/antitop_top.mp4'

    cap = cv2.VideoCapture(video_path)
    topStateDeque = TopStateDeque()
    armor_detector = ArmorDetector(cameraMatrix, distCoeffs, cameraVector)
    frame_num = 0  #当前是第几帧

    while True:
        success, frame = cap.read()
        if not success:
            break

        armors = armor_detector.detect(frame, 0, 0)  #检测出的装甲板
        antiTop = True
        useCamera = False

        frame_num += 1
        if antiTop:
            timeStampUs = cap.getTimeStampUs() if useCamera else int(time.time() * 1e6)
            topStateDeque.insertArmorsDetection2(armors,frame_num)
            topStateDeque.getTopState2()

        # 显示并绘制
        drawing = frame.copy()
        for a in armors:
            tools.drawContour(drawing, a.points)
            tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
            x, y, z = a.in_imu.T[0]
            tools.putText(drawing, f'x{x:.1f} y{y:.1f} z{z:.1f}', a.left.bottom, (255, 255, 255))

        cv2.imshow('press q to exit', drawing)

        key = cv2.waitKey(0) & 0xff
        if key == ord('q'):
            break

    cap.release()
