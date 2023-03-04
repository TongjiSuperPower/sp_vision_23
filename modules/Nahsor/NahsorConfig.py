
"""
参数配置
"""

# 形态学运算参数
CORE_SIZE = 10
OPEN_PARA = 1
CLOS_PARA = 1
DILA_PARA = 3   # 膨胀
EROD_PARA = 10   # 腐蚀

USE_HSV = False

# 使用预测
USE_PREDICT = True

# 目标颜色HSV
# H_RED_1 = (0, 180)
# H_RED_2 = (0, 10)
# S_RED = (0, 30)
# V_RED = (221, 255)

H_RED_1 = (170, 180)
H_RED_2 = (0, 10)
S_RED = (180, 255)
V_RED = (30, 255)

H_BLUE = (80, 124)
S_BLUE = (150, 255)
V_BLUE = (180, 255)

R_YUV_LOW = (12, 16, 160)
R_YUV_HIGH = (240, 150, 240)
B_YUV_LOW = (12, 140, 16)
B_YUV_HIGH = (240, 240, 100)

# 大小符
BIG = 1
SMALL = 0

# 转速列表判断为小符 所允许的最大方差
MAX_VAR = 5

# 大符转速范围，单位RPM
SPD_RANGE = (3, 25)

# 限制外接矩形的长宽比
ASPECT_RATIO = (1.4, 2.3)
ASPECT_RATIO_1 = (1.0, 2.3)

# 限制R标的长宽比
R_ASPECT_RATIO = (0.8, 1.5)

# 限制轮廓占外接矩形的面积比例
AREA_LW_RATIO = (0.3, 0.6)
AREA_LW_RATIO_1 = (0.5, 1)

# 目标装甲板占外接矩形的比例
ARMOR_AREA_RATIO = (0.03, 0.25)
ARMOR_AREA_RATIO_1 = (0.25, 0.7)

# 目标装甲板长宽比
ARMOR_WH_RATIO = (0.8, 2.5)

# 与能量机关的真实距离，单位m
DISTANCE = 7

# 弧度转角度
RAD2DEG = 57.3

# 装甲版的实际长和宽(单位:cm)
TARGET_WIDTH = 25.
TARGET_HEIGHT = 16.5
FAN_LEN = 64

# 焦距，单位px
FOCAL_LENGTH = 1200

# 轮廓面积最小值，用于筛除过小的轮廓，单位px
MIN_AREA = 150

# 最小误差
DBL_EPSILON = 2.2204e-016

# 相机与枪管的安装间距，单位m
BIAS = 0.05

# 前后两次识别出的R标位置 差异 的上限，单位 px
MIN_R_DIFF = 10

# 预测点与真实点位置 差异 的上限，单位 px
MIN_DIS_PRED = 10000

MIN_DIS = 50

# R标与目标装甲板的最大角度差
MAX_R_ANGLE = 2

# last_points的长度
LAST_P_LEN = 10

# 用于拟合圆的点的数目
FIT_C_LEN = 21

# 用于拟合正弦函数的点的数目
LAST_SPD_LEN = 50

INTERVAL = 0.05

REFIT_THRESH = 15

