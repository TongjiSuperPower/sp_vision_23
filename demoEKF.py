import cv2
import time
import numpy as np

import modules.tools as tools
from modules.armor_detection import ArmorDetector

if __name__ == '__main__':
    from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

    video_path = 'assets/antitop_top.mp4'
    # video_path = 'assets/input.avi'

    cap = cv2.VideoCapture(video_path)
    armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

    while True:
        success, frame = cap.read()
        if not success:
            break

        armors = armor_detector.detect(frame, 0, 0)

        drawing = frame.copy()
        for a in armors:
            tools.drawContour(drawing, a.points)
            tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
            x, y, z = a.in_imu.T[0]
            tools.putText(drawing, f'x{x:.1f} y{y:.1f} z{z:.1f}, yaw{a.yaw_in_imu}', a.left.bottom, (255, 255, 255))
            
        cv2.imshow('press q to exit', drawing)

        # 显示所有图案图片
        for i, a in enumerate(armor_detector._armors):
            cv2.imshow(f'{i}', a.pattern)

        key = cv2.waitKey(1) & 0xff
        if key == ord('q'):
            break

    cap.release()
