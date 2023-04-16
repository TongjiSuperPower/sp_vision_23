from math import cos, sin
import numpy as np
from enum import Enum

class TrackerState(Enum):
        LOST = 0
        DETECTING = 1
        TRACKING = 2
        TEMP_LOST = 3

class Tracker:
    '''单位采用m、s，弧度'''
    def __init__(self, max_match_distance, tracking_threshold, lost_threshold):
        self.tracker_state = TrackerState.LOST
        self.tracked_id = ""
        self.target_state = np.zeros((9,))
        self.max_match_distance_ = max_match_distance
        self.tracking_threshold_ = tracking_threshold
        self.lost_threshold_ = lost_threshold
        self.lost_count_ = 0
        self.detect_count_=0
        self.last_yaw_ = 0
        self.last_y_ = 0
        self.last_r_ = 0
        self.ekf = None

    def init(self, armors):
        # Simply choose the armor that is closest to image center
        tracked_armor = armors[0] # armors之前已经根据距离进行了排序，所以[0]就是最近的

        self.initEKF(tracked_armor)
        self.tracked_id = tracked_armor.name
        self.tracker_state = TrackerState.DETECTING

    def update(self, armors):
        # KF predict
        ekf_prediction = self.ekf.predict()
        
        print("EKF predict")

        matched = False
        # Use KF prediction as default target state if no matched armor is found
        target_state = ekf_prediction

        if len(armors)>0:
            min_position_diff = float("inf")
            predicted_position = self.getArmorPositionFromState(ekf_prediction)
            
            for armor in armors:                
                position_vec = np.reshape(armor.in_imuM,(3,))
                
                # Difference of the current armor position and tracked armor's predicted position
                position_diff = np.norm(predicted_position - position_vec)
                
                if position_diff < min_position_diff:
                    min_position_diff = position_diff
                    tracked_armor = armor
            
            if min_position_diff < self.max_match_distance_:
                # Matching armor found
                matched = True
                p = np.reshape(tracked_armor.in_imuM,(3,))
                # Update EKF
                measured_yaw = tracked_armor.yawR_in_imu
                z = np.array([p[0], p[1], p[2], measured_yaw])
                target_state = self.ekf.update(z)
                print("EKF update")
            else:
                # Check if there is same id armor in current frame
                for armor in armors:
                    if armor.name == self.tracked_id:
                        # Armor jump happens
                        matched = True
                        tracked_armor = armor
                        handleArmorJump(tracked_armor)
                        break

        # Suppress R from converging to zero
        if target_state[8] < 0.2:
            target_state[8] = 0.2
            self.ekf.setState(target_state)

        # Tracking state machine
        if self.tracker_state == TrackerState.DETECTING:
            if matched:
                self.detect_count_ += 1
                if self.detect_count_ > self.tracking_threshold_:
                    self.detect_count_ = 0
                    self.tracker_state = TrackerState.TRACKING
            else:
                self.detect_count_ = 0
                self.tracker_state = TrackerState.LOST
        elif self.tracker_state == TrackerState.TRACKING:
            if not matched:
                self.tracker_state = TrackerState.TEMP_LOST
                self.lost_count_ += 1
        elif self.tracker_state == TrackerState.TEMP_LOST:
            if not matched:
                self.lost_count_ += 1
                if self.lost_count_ > self.lost_threshold_:
                    self.lost_count_ = 0
                    self.tracker_state = TrackerState.LOST
            else:
                self.tracker_state = TrackerState.TRACKING
                self.lost_count_ = 0

    def initEKF(self, a):
        xa,ya,za = np.reshape(a.in_imuM,(3,))        
        self.last_yaw_ = 0
        yaw = a.yawR_in_imu

        # Set initial position at 0.2m behind the target
        target_state = np.zeros((9,))
        r = 0.2
        xc = xa + r * sin(yaw)
        yc = ya 
        zc = za + r * cos(yaw)
        self.last_y = yc
        self.last_r = r

        target_state[0] = xc
        target_state[1] = yc
        target_state[2] = zc
        target_state[3] = yaw
        target_state[8] = r

        self.ekf.setState(target_state)

        print("Init EKF!")
    
def handleArmorJump(self, a):
    self.last_yaw = self.target_state[3]
    yaw = a.yawR_in_imu

    if abs(yaw - self.last_yaw) > 0.4:
        self.last_y = self.target_state[2]
        self.target_state[2] = np.reshape(a.in_imuM,(3,))[2]
        self.target_state[3] = yaw
        self.target_state[8], last_r = last_r, self.target_state[8]
        print("Armor jump!")

    current_p = np.reshape(a.in_imuM,(3,))
    infer_p = self.getArmorPositionFromState(self.target_state)

    if np.norm(current_p - infer_p) > self.max_match_distance_:
        r = self.target_state[8]
        self.target_state[0] = current_p[0] + r * sin(yaw)
        self.target_state[2] = current_p[2] + r * cos(yaw)
        self.target_state[4] = 0
        self.target_state[6] = 0
        print("State wrong!")

    self.ekf.setState(self.target_state)


def getArmorPositionFromState(self, x):
    # Calculate predicted position of the current armor
    xc, yc, zc = x[:3]
    yaw, r = x[3], x[8]
    xa = xc - r * sin(yaw)
    za = zc - r * cos(yaw)
    return np.array([xa, yc, za])