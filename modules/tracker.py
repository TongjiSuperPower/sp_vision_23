import math
import numpy as np
from enum import IntEnum
from modules.NewEKF import ExtendedKalmanFilter
from modules.autoaim.armor import Armor
from modules.tools import shortest_angular_distance
from modules.target import NormalRobot, BalanceInfantry, Outpost, Base


class TrackerState(IntEnum):
    LOST = 0
    DETECTING = 1
    TRACKING = 2
    TEMP_LOST = 3


class Tracker:
    '''单位采用m、s，弧度'''

    def __init__(self, max_match_distance, tracking_threshold, lost_threshold):
        self.tracking_target = None
        self.max_match_distance = max_match_distance
        self.tracking_threshold = tracking_threshold
        self.lost_threshold = lost_threshold

        self.tracker_state = TrackerState.LOST

        self.lost_count = 0
        self.detect_count = 0

    def init(self, armors: list[Armor]):
        # 进入LOST状态后，必须要检测到装甲板才能初始化tracker
        if len(armors) == 0:
            return

        # 选择最近的装甲板作为目标
        armor = armors[0]

        # 根据装甲板id调用相应的目标模型
        if (armor.name in ("big_three", "big_four", "big_five")):
            self.tracking_target = BalanceInfantry()
        # elif (armor.name == "small_outpost"):
        #     self.tracking_target = Outpost()
        elif (armor.name == "small_base"):
            self.tracking_target = Base()
        else:
            self.tracking_target = NormalRobot()

        # 初始化目标模型
        self.tracking_target.init(armor)

        self.tracker_state = TrackerState.DETECTING

    def update(self, armors: list[Armor], dt):
        self.tracking_target.arrmor_jump = 0
        self.tracking_target.state_error = 0

        # predict
        prediction = self.tracking_target.forwardPredict(dt)

        matched = False

        # Use KF prediction as default target state if no matched armor is found
        self.tracking_target.setTargetState(prediction)

        if len(armors) > 0:
            min_position_diff = float('inf')
            predicted_position = self.tracking_target.getArmorPositionFromState(prediction)

            matched_armor = None
            for armor in armors:
                position_vec = np.reshape(armor.in_imu_m, (3,))

                position_diff = np.linalg.norm(predicted_position - position_vec)

                if position_diff < min_position_diff:
                    min_position_diff = position_diff
                    matched_armor = armor

            if min_position_diff < self.max_match_distance:
                matched = True

                # Update
                self.tracking_target.update(matched_armor)

            else:
                # Check if there is same id armor in current frame
                for armor in armors:
                    if armor.name == self.tracking_target.armor_id:
                        # Armor jump happens
                        matched = True
                        matched_armor = armor

                        self.tracking_target.handleArmorJump(matched_armor, self.max_match_distance)

                        break

        self.tracking_target.limitStateValue()

        # Tracking state machine
        if self.tracker_state == TrackerState.DETECTING:
            if matched:
                self.detect_count += 1
                if self.detect_count > self.tracking_threshold:
                    self.detect_count = 0
                    self.tracker_state = TrackerState.TRACKING
                    print("tracker start tracking")
            else:
                self.detect_count = 0
                self.tracker_state = TrackerState.LOST

        elif self.tracker_state == TrackerState.TRACKING:
            if not matched:
                self.tracker_state = TrackerState.TEMP_LOST
                self.lost_count += 1

        elif self.tracker_state == TrackerState.TEMP_LOST:
            if not matched:
                self.lost_count += 1
                if self.lost_count > self.lost_threshold:
                    self.lost_count = 0
                    self.tracker_state = TrackerState.LOST
                    print("tracker has lost")
            else:
                self.tracker_state = TrackerState.TRACKING
                self.lost_count = 0

    def getShotPoint(self, deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0):
        '''获取预测时间后待击打点的位置(单位:mm)(无重力补偿)'''
        return self.tracking_target.getPreShotPtsInImu(deltatime, bulletSpeed, R_camera2gimbal, t_camera2gimbal, cameraMatrix, distCoeffs, yaw=0, pitch=0)
