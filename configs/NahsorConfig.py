# coding:utf-8
"""
参数配置
"""
from enum import Enum


# 状态参数
class STATUS(Enum):
    NOT_FOUND = 0
    FOUND = 1


class FIND_TARGET_MODE(Enum):
    MIDDLE = 2
    FAR = 3


class FIT_STATUS(Enum):
    FAILED = 0
    FITTING = 1
    SUCCESS = 2


# 拟合速度函数的方法参数
class FIT_SPEED_MODE(Enum):
    CURVE_FIT = 1
    PARAM_FIT = 2


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

# 使用预测
USE_PREDICT = True

# 目标颜色设置合集
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
ARMOR_AREA_RATIO = (0.15, 0.29)
# ARMOR_AREA_RATIO_1 = (0.25, 0.7)

# 目标装甲板长宽比
ARMOR_WH_RATIO = (0.8, 2.5)
# 正方形长宽比
SQUARE_WH_RATIO = (0.9, 1.1)
# 装甲板上半部分长宽比
UPPER_WH_RATIO = (0.9, 1.2)
# 能量机关中心与目标中心距离和目标半径的比值
CENTER_DISTANCE_RATIO = (2.5, 4.2)

# # 装甲版的实际长和宽(单位:cm)
# TARGET_WIDTH = 25.
# TARGET_HEIGHT = 16.5
# FAN_LEN = 64

# # 焦距，单位px
# FOCAL_LENGTH = 1200

# 轮廓面积最小值，用于筛除过小的轮廓，单位px
# 注意过小时可能会把r标滤掉
CONTOUR_MIN_AREA = 500
# R标大小范围
R_AREA_RANGE = (200, 450)

# 最小误差
DBL_EPSILON = 2.2204e-016

# 目标两帧间最大距离差
CENTER_MAX_DISTANCE = 50
# R最大距离差
R_MAX_DISTANCE = 20

# 用于拟合圆的点的数目
TARGET_CENTERS_LEN = 50

# 用于拟合正弦函数的点的数目
FIT_MAX_LEN = 100
FIT_MIN_LEN = 5

# 重新拟合的时间间隔 单位为s
FIND_R_INTERVAL = 5
SPEED_REFIT_INTERVAL = 2
# 每个拟合点的采样间隔
FIT_INTERVAL = 0.2
# # 拟合最大误差
# FIT_MAX_ERROR = 0.1
# 拟合正弦函数的参数上下限
SPEED_PARAM_BOUNDS = {
    # "a": (0.780, 1.045),
    # "w": (1.884, 2.000),
    # "b": (2.090 - 1.045, 2.090 - 0.780)

    "a": [0.50, 1.2],
    "w": [1.6, 2.200],
    "b": [0.8, 1.3]
}
SPEED_PARAM_MAXERROR = {
    "a": 0.05,
    "w": 0.05,
    "b": 0.05
}
# 小符旋转速度  RPM
SMALL_ROT_SPEED = 16
# 延迟时间
DELAY_TIME = 0.50
