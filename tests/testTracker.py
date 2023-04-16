import cv2
import time
import numpy as np
from collections import deque

import modules.tools as tools
from modules.io.robot import Robot
from modules.ExtendKF import EKF
from modules.armor_detection import ArmorDetector
from modules.tracker import Tracker
from modules.NewEKF import ExtendedKalmanFilter

from remote_visualizer import Visualizer


# EKF
# xa = x_armor, xc = x_robot_center
# state: xc, yc, zc, yaw, v_xc, v_yc, v_zc, v_yaw, r
# measurement: xa, ya, za, yaw
# f - Process function
def f(x, dt):
    x_new = np.copy(x)
    x_new[0] += x[4] * dt
    x_new[1] += x[5] * dt
    x_new[2] += x[6] * dt
    x_new[3] += x[7] * dt
    return x_new

# J_f - Jacobian of process function


def j_f(x, dt):
    dfdx = np.zeros((9, 9))
    dfdx[0, 0] = dfdx[1, 1] = dfdx[2, 2] = dfdx[3, 3] = 1
    dfdx[0, 4] = dt
    dfdx[1, 5] = dt
    dfdx[2, 6] = dt
    dfdx[3, 7] = dt
    return dfdx

# h - Observation function


def h(x):
    z = np.zeros(4)
    xc, yc, yaw, r = x[0], x[1], x[3], x[8]
    z[0] = xc - r * np.cos(yaw)  # xa
    z[1] = yc - r * np.sin(yaw)  # ya
    z[2] = x[2]                 # za
    z[3] = yaw                  # yaw
    return z

# J_h - Jacobian of observation function


def j_h(x):
    dhdx = np.zeros((4, 9))
    yaw, r = x[3], x[8]
    dhdx[0, 0] = dhdx[1, 1] = dhdx[2, 2] = dhdx[3, 3] = 1
    dhdx[0, 3] = -r * np.sin(yaw)
    dhdx[1, 3] = r * np.cos(yaw)
    dhdx[0, 8] = -np.cos(yaw)
    dhdx[1, 8] = -np.sin(yaw)
    return dhdx

class Target_msg:
    def __init__(self):
        self.position = [0.0, 0.0, 0.0]
        self.yaw = 0.0
        self.velocity = [0.0, 0.0, 0.0]
        self.v_yaw = 0.0
        self.radius_1 = 0.0
        self.radius_2 = 0.0
        self.z_2 = 0.0


if __name__ == '__main__':
    with Robot(3, '/dev/ttyUSB0') as robot, Visualizer(enable=False) as visualizer:

        robot.update()
        if robot.id == 3:
            from configs.infantry3 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal
        elif robot.id == 4:
            from configs.infantry4 import cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal

        armor_detector = ArmorDetector(
            cameraMatrix, distCoeffs, R_camera2gimbal, t_camera2gimbal)

        twoPtsInCam = deque(maxlen=2)  # a queue with max 2 capaticity
        twoPtsInWorld = deque(maxlen=2)
        twoPtsInTripod = deque(maxlen=2)
        twoTimeStampMs = deque(maxlen=2)
        continuousFrameCount = 0

        while True:
            robot.update()

            img = robot.img
            armors = armor_detector.detect(img, robot.yaw, robot.pitch)
            armors = list(filter(lambda a: a.color != robot.color, armors))
            armors.sort(key=lambda a: a.observation[0])  # 优先击打最近的

            twoTimeStampMs.append(robot.camera_stamp_ms)
            deltaTime = (
                twoTimeStampMs[1] - twoTimeStampMs[0]) if len(twoTimeStampMs) == 2 else 5
            deltaTime = deltaTime/1000 # 转换为s

            # 调试用，绘图
            drawing = img.copy()
            for i, l in enumerate(armor_detector._filtered_lightbars):
                tools.drawContour(drawing, l.points, (0, 255, 255), 10)
            for i, lp in enumerate(armor_detector._filtered_lightbar_pairs):
                tools.drawContour(drawing, lp.points)
            for i, a in enumerate(armors):
                cx, cy, cz = a.in_camera.T[0]
                tools.drawAxis(drawing, a.center, a.rvec,
                               a.tvec, cameraMatrix, distCoeffs)
                tools.putText(
                    drawing, f'{i} {a.color} {a.name} {a.confidence:.2f}', a.left.top, (255, 255, 255))
                tools.putText(
                    drawing, f'cx{cx:.1f} cy{cy:.1f} cz{cz:.1f}', a.left.bottom, (255, 255, 255))
            visualizer.show(drawing)

            # EKF整车观测：
            # Tracker
            max_match_distance = 0.2  # 单位:m
            tracking_threshold = 5  # 从检测到->跟踪的帧数
            lost_threshold = 5  # 从暂时丢失->丢失的帧数
            tracker = Tracker(max_match_distance,
                              tracking_threshold, lost_threshold)

            # EKF参数
            # Q - process noise covariance matrix
            q_v = [1e-2, 1e-2, 1e-2, 2e-2, 5e-2, 5e-2, 1e-4, 4e-2, 1e-3]
            Q = np.diag(q_v)
            # R
            r_v = [1e-1, 1e-1, 1e-1, 2e-1]
            R = np.diag(r_v)
            # P - error estimate covariance matrix
            P0 = np.eye(9)

            tracker.ekf = ExtendedKalmanFilter(f, h, j_f, j_h, Q, R, P0)

            if tracker.tracker_state == Tracker.LOST:
                if len(armors)==0:
                    continue

                # 进入LOST状态后，必须要检测到装甲板才能初始化tracker
                tracker.init(armors)
                
            else:
               
                tracker.update(armors)


            state = tracker.target_state

            msg = Target_msg()
            msg.position = [state[0], state[1], state[2]]
            msg.yaw = state[3]
            msg.velocity = [state[4], state[5], state[6]]
            msg.v_yaw = state[7]
            msg.radius_1 = state[8]
            msg.radius_2 = tracker.last_r
            msg.z_2 = tracker.last_z      

            robot.send(ppx, ppy-60, ppz)

            # 调试用
            # visualizer.plot((cy, y, robot.yaw*10, robot.pitch*10), ('cy', 'y', 'yaw', 'pitch'))
            visualizer.plot((x, y, z, robot.yaw*10, robot.pitch*10),
                            ('x', 'y', 'z', 'yaw', 'pitch'))
            # visualizer.plot((x, y, z, px, py, pz), ('x', 'y', 'z', 'px', 'py', 'pz'))
            # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz'))
            # visualizer.plot((x, y, z, px, py, pz, ppx, ppy, ppz,vx*10,vy*10,vz*10), ('x', 'y', 'z', 'px', 'py', 'pz','ppx','ppy','ppz','vx','vy','vz'))
