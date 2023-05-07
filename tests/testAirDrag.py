import os
import sys

tests_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(tests_folder)
sys.path.append(parent_folder)
import modules.tools as tools


import cv2
import math
import numpy as np
from queue import Empty
from multiprocessing import Queue
from typing import Tuple
from scipy.integrate import solve_ivp

class Robot:
    def __init__(self, exposure_ms: float, port: str) -> None:
        self.bullet_speed=0
        self.id=""


robot = Robot(3,'/dev/ttyUSB0')
robot.id = 'big_one'
robot.bullet_speed = 15
pitch_offset = 0
predictedPtsInWorld = [243.23, 873.45, 5109.56]
import time
# 记录开始时间
start_time = time.time()

# 程序代码
for i in range(10):
    armor_in_gun = tools.trajectoryAdjust(predictedPtsInWorld, 
                                      pitch_offset, 
                                      robot, 
                                      enableAirRes=1)

# 记录结束时间
end_time = time.time()

# 计算运行时间
run_time = (end_time - start_time)*1000

print("程序运行时间为：", run_time, "ms")
print(armor_in_gun)
