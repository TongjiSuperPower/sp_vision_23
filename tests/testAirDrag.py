import os
import sys

tests_folder = os.path.dirname(__file__)
parent_folder = os.path.dirname(tests_folder)
sys.path.append(parent_folder)
from modules.io.robot import Robot

import cv2
import math
import numpy as np
from queue import Empty
from multiprocessing import Queue
from typing import Tuple
from scipy.integrate import solve_ivp



def shoot_pitch(x, y, z, bullet_speed) -> float:
    g = 9.794 / 1000
    distance = (x**2 + z**2)**0.5

    a = 0.5 * g * distance**2 / bullet_speed**2
    b = -distance
    c = a - y

    result1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
    result2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)
    pitch1 = math.atan(result1)
    pitch2 = math.atan(result2)
    t1 = distance / (bullet_speed * math.cos(pitch1))
    t2 = distance / (bullet_speed * math.cos(pitch2))

    pitch = pitch1 if t1 < t2 else pitch2
    pitch = math.degrees(pitch)

    return pitch

def trajectoryAdjust(target_pos, pitch_offset, robot:Robot, enableAirRes=1):
    '''
    弹道调整，返回重力（及空气阻力）补偿后的目标位置。

    target_pos: mm;
    pitch_offset: 度;
    robot: 本机;
    enableAirRes: 是否计算空气阻力。

    return: mm。
    '''
    pos = np.reshape(target_pos, (3,))
    x, y, z = pos
    pitch = shoot_pitch(x, y, z, robot.bullet_speed) + pitch_offset # 枪管向上抬为正
    
    if enableAirRes==1:
        try:
            # Drag coefficient, projectile radius (m), area (m2) and mass (kg).
            c = 0.47
            r = 21/1000 if robot.id=='big_one' else 8.5/1000
            A = np.pi * r**2
            m = 0.04 if robot.id=='big_one' else 0.0032
            # Air density (kg.m-3), acceleration due to gravity (m.s-2).
            rho_air = 1.28
            g = 9.81
            # For convenience, define  this constant.
            k = 0.5 * c * rho_air * A
            pitch = findPitch(robot.bullet_speed, k, m, g, math.sqrt(x**2+z**2), y, pitch-5, pitch+10)
            pitch += pitch_offset
        except:
            print("弹道空气阻力补偿计算出错")
    
    armor_in_gun = np.array([x, y, z]).reshape(3, 1)
    armor_in_gun[1] = (x*x + z*z) ** 0.5 * -math.tan(math.radians(pitch))

    return armor_in_gun

def calculateDrop(bulletSpeed, k, m, g, pitch, distance):
    '''
    计算弹丸在空气阻力和重力的作用下下坠距离,返回y坐标(mm)
    
    弹速(m/s); 空气阻力系数; 质量(kg); 重力加速度(m/s2); pitch(度); 水平飞行距离(mm);
    '''
    distanceM = distance/1000 # mm -> m

    v0 = bulletSpeed
    phi0 = np.radians(pitch)   
    u0 = 0, v0 * np.cos(phi0), 0., v0 * np.sin(phi0)
    t0, tf = 0, 10

    def deriv(t, u):
        x, xdot, z, zdot = u
        speed = np.hypot(xdot, zdot)
        xdotdot = -k/m * speed * xdot
        zdotdot = -k/m * speed * zdot - g
        return xdot, xdotdot, zdot, zdotdot

    def hit_target(t, u):
        # We've hit the target if the z-coordinate is 0.
        return u[0]-distanceM
    
    # Stop the integration when we hit the target.
    hit_target.terminal = True
    # We must be moving downwards (don't stop before we begin moving upwards!)
    hit_target.direction = 1

    soln = solve_ivp(deriv, (t0, tf), u0, dense_output=False,
                 events=(hit_target))
    
    y = soln.y_events[0][0][2]
    y = -y*1000 # m -> mm

    return y

def findPitch(bulletSpeed, k, m, g, distance, y, x0, x1, tol=1e-6, maxiter=100):
    '''割线法求解'''
    f0, f1 = y - calculateDrop(bulletSpeed, k, m, g, x0, distance), y - calculateDrop(bulletSpeed, k, m, g, x1, distance)

    for i in range(maxiter):
        if abs(f1) < tol:
            return x1
        dfdx = (f1 - f0) / (x1 - x0)
        x2 = x1 - f1 / dfdx
        f0, f1 = f1, y - calculateDrop(bulletSpeed, k, m, g, x2, distance)
        x0, x1 = x1, x2

    raise RuntimeError('Failed to converge')


robot = Robot(3,'/dev/ttyUSB0')
robot.id = 'big_one'
robot.bullet_speed = 15
pitch_offset = 0
predictedPtsInWorld = [243.23, 873.45, 5109.56]
import time
# 记录开始时间
start_time = time.time()

# 程序代码
for i in range(1):
    armor_in_gun = trajectoryAdjust(predictedPtsInWorld, 
                                      pitch_offset, 
                                      robot, 
                                      enableAirRes=0)

# 记录结束时间
end_time = time.time()

# 计算运行时间
run_time = (end_time - start_time)

print("程序运行时间为：", run_time, "ms")
print(armor_in_gun)
