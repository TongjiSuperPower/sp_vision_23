# coding:utf-8
"""
参数配置
"""
from enum import Enum

import numpy as np


# 状态参数
class TARGET_STATUS(Enum):
    NOT_FOUND = 0
    FOUND = 1


class FIT_SPEED_STATUS(Enum):
    FAILED = 0
    FITTING = 1
    SUCCESS = 2



# 拟合速度函数的方法参数
class FIT_SPEED_MODE(Enum):
    BY_SPEED = 1
    BY_ANGLE = 2


class COLOR(Enum):
    RED = 'RED'
    BLUE = 'BLUE'


class COLOR_SPACE(Enum):
    HSV = 'HSV'
    BGR = 'BGR'
    YUV = 'YUV'
    SIMU_BLACK = 'SIMU_BLACK'


class ENERGY_MODE(Enum):
    # 大小符
    BIG = 1
    SMALL = 0


# 形态学运算参数
CORE_SIZE = 1
OPEN_PARA = 1
CLOS_PARA = 1
DILA_PARA = 3  # 膨胀
EROD_PARA = 10  # 腐蚀

USE_HSV = False

# 使用预测
USE_PREDICT = True

# 目标颜色HSV
# H_RED_1 = (0, 180)
# H_RED_2 = (0, 10)
# S_RED = (0, 30)
# V_RED = (221, 255)
HSV_RED_UPPER_1 = (180, 255, 255)
HSV_RED_LOWER_1 = (170, 180, 30)
HSV_RED_UPPER_2 = (10, 255, 255)
HSV_RED_LOWER_2 = (0, 180, 30)

HSV_BLUE_UPPER = (124, 255, 255)
HSV_BLUE_LOWER = (80, 150, 180)

YUV_RED_UPPER = (240, 150, 240)
YUV_RED_LOWER = (12, 16, 160)
YUV_BLUE_UPPER = (240, 240, 100)
YUV_BLUE_LOWER = (12, 140, 16)

BGR_RED_UPPER = (200, 200, 250)
BGR_RED_LOWER = (70, 70, 130)
BGR_BLUE_UPPER = (255, 200, 110)
BGR_BLUE_LOWER = (240, 140, 0)

# # unity录屏+黑化处理
SIMU_BLACK_RED_UPPER = (50, 70, 255)
SIMU_BLACK_RED_LOWER = (0, 0, 230)

# 转速列表判断为小符 所允许的最大方差
# MAX_VAR = 5

# 大符转速范围，单位RPM
# SPD_RANGE = (3, 25)

# 限制外接矩形的长宽比
# ASPECT_RATIO = (1.4, 2.3)
# ASPECT_RATIO_1 = (1.0, 2.3)

# 限制R标的长宽比
# R_ASPECT_RATIO = (0.8, 1.5)

# 限制轮廓占外接矩形的面积比例
# AREA_LW_RATIO = (0.3, 0.6)
# AREA_LW_RATIO_1 = (0.5, 1)

# 目标装甲板占外接矩形的比例
# ARMOR_AREA_RATIO = (0.03, 0.25)
# ARMOR_AREA_RATIO_1 = (0.25, 0.7)

# 目标装甲板长宽比
ARMOR_WH_RATIO = (0.8, 2.5)
# 正方形长宽比
SQUARE_WH_RATIO = (0.9, 1.1)

# 能量机关中心与目标中心距离和目标半径的比值
CENTER_DISTANCE_RATIO = (3.8, 4.2)

# 与能量机关的真实距离，单位m
# DISTANCE = 7

# 弧度转角度
RAD2DEG = 57.3

# 装甲版的实际长和宽(单位:cm)
TARGET_WIDTH = 25.
TARGET_HEIGHT = 16.5
FAN_LEN = 64

# 焦距，单位px
FOCAL_LENGTH = 1200

# 轮廓面积最小值，用于筛除过小的轮廓，单位px
CONTOUR_MIN_AREA = 50

# 最小误差
DBL_EPSILON = 2.2204e-016

# 相机与枪管的安装间距，单位m
# BIAS = 0.05

# 前后两次识别出的R标位置 差异 的上限，单位 px
# MIN_R_DIFF = 10

# 预测点与真实点位置 差异 的上限，单位 px
# MIN_DIS_PRED = 10000
# 目标最大距离差
CENTER_MAX_DISTANCE = 50
# R最大距离差
R_MAX_DISTANCE = 20

# # R标与目标装甲板的最大角度差
# MAX_R_ANGLE = 2

# # last_points的长度
# LAST_P_LEN = 10

# 用于拟合圆的点的数目
TARGET_CENTERS_LEN = 50

# 用于拟合正弦函数的点的数目
FIT_MAX_LEN = 100
FIT_MIN_LEN = 5

# 重新拟合的时间间隔 单位为s
R_REFIT_INTERVAL = 5
SPEED_REFIT_INTERVAL = 5
# 每个拟合点的采样间隔
FIT_INTERVAL = 0.2
# 拟合最大误差
FIT_MAX_ERROR = 0.1
# 拟合正弦函数的参数上下限
SPEED_PARAMS = {
    "a": (0.780, 1.045),
    "w": (1.884, 2.000),
    "b": (2.090 - 1.045, 2.090 - 0.780)
}
# LAST_SPD_LEN = 50

INTERVAL = 0.05

REFIT_THRESH = 15

# 延迟时间
DELAY_TIME = 0.50
