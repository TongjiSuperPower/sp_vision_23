import cv2
import math
import time
import numpy as np

import modules.tools as tools
from modules.armor_detection import ArmorDetector
from modules.tracker import Tracker, TrackerState
from modules.NewEKF import ExtendedKalmanFilter

from remote_visualizer import Visualizer


class Target_msg:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.v_yaw = 0.0
        self.radius_1 = 0.0
        self.radius_2 = 0.0
        self.y_2 = 0.0


if __name__ == '__main__':
    from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

    video_path = 'assets/antitop_top.mp4'
    # video_path = 'assets/input.avi'

    cap = cv2.VideoCapture(video_path)
    armor_detector = ArmorDetector(cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

    # EKF整车观测：
    # Tracker
    max_match_distance = 0.2  # 单位:m
    tracking_threshold = 5  # 从检测到->跟踪的帧数
    lost_threshold = 5  # 从暂时丢失->丢失的帧数
    tracker = Tracker(max_match_distance, tracking_threshold, lost_threshold)

    while True:
        success, frame = cap.read()
        if not success:
            break

        armors = armor_detector.detect(frame, 0, 0)
        armors.sort(key=lambda a: a.observation[0]) # 优先击打最近的

        if tracker.tracker_state == TrackerState.LOST:
            # 进入LOST状态后，必须要检测到装甲板才能初始化tracker
            if len(armors) == 0:
                print('lost.')
                continue

            tracker.init(armors)

        else:
            tracker.update(armors, 1/60)

        state = tracker.target_state

        msg = Target_msg()
        msg.position = [state[0], state[1], state[2]]
        msg.yaw = state[3]
        msg.velocity = [state[4], state[5], state[6]]
        msg.v_yaw = state[7]
        msg.radius_1 = state[8]
        msg.radius_2 = tracker.last_r
        msg.y_2 = tracker.last_y

        # 以下为调试相关的代码

        drawing = frame.copy()
        for a in armors:
            tools.drawContour(drawing, a.points)
            tools.drawAxis(drawing, a.center, a.rvec, a.tvec, cameraMatrix, distCoeffs)
            tools.putText(drawing, f'{a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
            x, y, z = a.in_imu.T[0]
            tools.putText(drawing, f'rotate speed: {msg.v_yaw:.2f}', (100, 100), (255, 255, 255))
            tools.putText(drawing, f'radius1: {msg.radius_1:.2f} radius2: {msg.radius_2:.2f}', (100, 200), (255, 255, 255))

        cv2.imshow('press q to exit', drawing)

        key = cv2.waitKey(16) & 0xff
        if key == ord('q'):
            break

    cap.release()
