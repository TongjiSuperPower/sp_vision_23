#!/usr/bin/env python
#

import numpy as np
import collections
from modules.Nahsor.NahsorConfig import *
import pywt
from scipy import optimize
import matplotlib.pyplot as plt

"""
相关计算函数
"""


def calc_point_angle(p1, p2):
    """
    计算两点连线的角度 / x轴逆时针需旋转多少角度到该位置
    :param p1:
    :param p2:
    :return: 角度 0～360
    """
    # if p1[0] > p2[0]:
    #     p1, p2 = p2, p1

    radian = np.arctan2(p2[1] - p1[1], p2[0] - p1[0])
    angle = radian * 180 / np.pi

    return angle


def calc_distance(p1, p2):
    """
    计算点p1, p2的距离
    :param p1:
    :param p2:
    :return: 距离: float
    """
    if p1 is None or p2 is None:
        return np.inf
    dis_sq = (p1[0] - p2[0]) * (p1[0] - p2[0]) + (p1[1] - p2[1]) * (p1[1] - p2[1])
    return np.sqrt(dis_sq)


def calc_angle(p1, p2, c):
    """
    计算从p1到p2、以c为圆心，需要转动的角度
    :param p1: 起始点
    :param p2: 终止点
    :param c: 圆心
    :return: 角度 0～360
    """
    res = abs(calc_point_angle(p2, c) - calc_point_angle(p1, c))
    if res > 300:
        res = 360 - res
    return res


def get_distance(width, w_scale):
    """
    返回所在点与装甲板目标的距离，单位：m
    """
    dis = ((TARGET_WIDTH / w_scale) * FOCAL_LENGTH) / (width * 100)
    return "%.2fm" % dis


def get_rotate_direction(points, section):
    """
    返回大符的转动方向
    """
    x1, x2, y1, y2 = points[0][0], points[1][0], points[0][1], points[1][1]
    if (section == 1 and x2 >= x1 and y2 >= y1) or \
            (section == 2 and x2 >= x1 and y2 <= y1) or \
            (section == 3 and x2 <= x1 and y2 <= y1) or \
            (section == 4 and x2 <= x1 and y2 >= y1):
        return 0
    else:
        return 1


def get_section(c_center, p1, p2):
    """
    返回p1,p2所在象限，若不在同一象限，返回-1
    """
    if (p1[0] > c_center[0] and p2[0] > c_center[0]) \
            and (p1[1] < c_center[1] and p2[1] < c_center[1]):
        return 1
    elif (p1[0] < c_center[0] and p2[0] < c_center[0]) \
            and (p1[1] < c_center[1] and p2[1] < c_center[1]):
        return 2
    elif (p1[0] < c_center[0] and p2[0] < c_center[0]) \
            and (p1[1] > c_center[1] and p2[1] > c_center[1]):
        return 3
    elif (p1[0] > c_center[0] and p2[0] > c_center[0]) \
            and (p1[1] > c_center[1] and p2[1] > c_center[1]):
        return 4
    else:
        return -1


def get_model(w_scale, h_scale):
    """
    按序返回矩形的四个顶点的真实坐标值，用于PnP解算
    :return: np.ndarray, 四个顶点，按从左上开始的顺时针顺序
    """
    w = TARGET_WIDTH / w_scale
    h = TARGET_HEIGHT / h_scale
    return np.array([[0., 0., 0.], [-w / 2, h / 2, 0.], [w / 2, h / 2, 0.],
                     [w / 2, -h / 2, 0.], [-w / 2, -h / 2, 0.],
                     ], dtype=float)


def get_vertex(box, rect):
    """
    :param box:
    :param rect:
    :return: 按序排列的四个顶点的图像上的坐标
    """
    if rect[1][0] > rect[1][1]:
        box = [box[1], box[2], box[3], box[0]]
    return np.array(box, dtype=int)


def get_position(pre_point, center_point, distance, px2m):
    x = (pre_point[0] - center_point[0]) * px2m * 100
    y = -(pre_point[1] - center_point[1]) * px2m * 100
    return np.array([x, y, distance * 100])


def get_spd_by_equation(t):
    """
    转动方程，以速度最低点为初相
    :param t: 时间间隔
    :return: 转速
    """
    return 9.55 * (0.785 * np.sin(1.884 * (t + 2.5)) + 1.305)


def get_spd_by_smoothing(last_spds, spd):
    """
    根据last_spds的趋势推断当前速度spd的合理性并修正
    :param last_spds: 前20次测量的速度
    :param spd: 当前速度
    :return: 修正后的当前速度
    """
    pass


def color_list():
    """
    定义字典存放颜色分量上下限
    例如：{颜色: [min分量, max分量]}
    'red': [array([160,  43,  46]), array([179, 255, 255])]}
    """
    _dict = collections.defaultdict(list)

    # 红色
    lower_red = np.array([H_RED_1[0], S_RED[0], V_RED[0]])
    upper_red = np.array([H_RED_1[1], S_RED[1], V_RED[1]])
    color_list = [lower_red, upper_red]
    _dict['red'] = color_list

    # 红色2
    lower_red = np.array([H_RED_2[0], S_RED[0], V_RED[0]])
    upper_red = np.array([H_RED_2[1], S_RED[1], V_RED[1]])
    color_list = [lower_red, upper_red]
    _dict['red2'] = color_list

    # 蓝色
    lower_blue = np.array([H_BLUE[0], S_BLUE[0], V_BLUE[0]])
    upper_blue = np.array([H_BLUE[1], S_BLUE[1], V_BLUE[1]])
    color_list = [lower_blue, upper_blue]
    _dict['blue'] = color_list

    return _dict


def get_r_debug(rect_armor, rect_parent, cur_point):
    theta = rect_armor[-1]
    if abs(rect_armor[-1] - 0) < DBL_EPSILON:
        theta = 0
    a = (90 - theta) / RAD2DEG
    b = theta / RAD2DEG
    radius = 5.5 * min(rect_armor[1][0], rect_armor[1][1])
    r_center = None
    # 判断子轮廓与父轮廓的相对位置，进而判断当前装甲板的相对位置
    x1, y1, x2, y2 = rect_armor[0][0], rect_armor[0][1], rect_parent[0][0], rect_parent[0][1]

    if x1 - x2 >= 0 and y1 - y2 <= 0:
        # 第一象限
        r_center = (int(cur_point[0] - radius * np.cos(a)),
                    int(cur_point[1] + radius * np.sin(a)))
    elif x1 - x2 >= 0 and y1 - y2 >= 0:
        # 第二象限
        r_center = (int(cur_point[0] - radius * np.cos(b)),
                    int(cur_point[1] - radius * np.sin(b)))
    elif x1 - x2 <= 0 and y1 - y2 >= 0:
        # 第三象限
        r_center = (int(cur_point[0] + radius * np.cos(a)),
                    int(cur_point[1] - radius * np.sin(a)))
    elif x1 - x2 <= 0 and y1 - y2 <= 0:
        # 第四象限
        r_center = (int(cur_point[0] + radius * np.cos(b)),
                    int(cur_point[1] + radius * np.sin(b)))

    return radius, r_center


def get_r_by_position(rect_armor, rect_parent, R_edge, cur_point):
    """
    获取旋转圆心R的坐标
    :param rect_armor:
    :param rect_parent:
    :param R_edge:
    :param cur_point:
    :return:
    """
    pLen = max(rect_parent[1][0], rect_parent[1][1])
    p_center = rect_parent[0]

    # 判断离父轮廓距离合适的小轮廓
    for R in R_edge:
        tmp_r_center = R[1][0]

        # 限制R到父轮廓的距离
        if 0.65 <= (calc_distance(tmp_r_center, p_center) / pLen) <= 1.3:
            # 限制R与父轮廓中心的连线角度
            sim = abs(calc_point_angle(p_center, tmp_r_center) - rect_armor[-1])
            if sim <= MAX_R_ANGLE or abs(sim - 90) or abs(sim - 180) <= MAX_R_ANGLE:
                r_center = (int(tmp_r_center[0]), int(tmp_r_center[1]))
                radius = int(calc_distance(r_center, cur_point))
                return radius, r_center
    return [None] * 2


def get_mid_pnts(rect, box):
    center = rect[0]
    left_mid, right_mid = [None] * 2
    mids = []
    for i in range(len(box)):
        for j in range(i + 1, len(box)):
            mids.append(0.5 * (box[i] + box[j]))
    w, h = max(rect[1][0], rect[1][1]), min(rect[1][0], rect[1][1])
    for m in mids:
        if -2 <= calc_distance(m, center) - w / 2 <= 2:
            if (m[0] - center[0]) < -2:
                left_mid = m
            elif (m[0] - center[0]) > 2:
                right_mid = m
            else:
                if abs(m[1] - center[1]) > 2:
                    if (m[1] - center[1]) < 2:
                        left_mid = m
                    elif (m[1] - center[1]) > 2:
                        right_mid = m
    return left_mid, right_mid


def get_r_by_circle(points):
    """
        通过最小二乘法来拟合圆的信息
        参数：self.cirV: 所有点坐标
        返回值: center: 得到的圆心坐标
               radius: 圆的半径
        """

    if len(points) < 3:
        return [None] * 2

    [sumX, sumY, sumX2, sumY2, sumX3, sumY3, sumXY, sumX1Y2, sumX2Y1] = [0.0] * 9
    N = len(points)
    for i in range(len(points)):
        x, y = points[i][0], points[i][1]
        x2 = x * x
        y2 = y * y
        x3 = x2 * x
        y3 = y2 * y
        xy = x * y
        x1y2 = x * y2
        x2y1 = x2 * y

        sumX += x
        sumY += y
        sumX2 += x2
        sumY2 += y2
        sumX3 += x3
        sumY3 += y3
        sumXY += xy
        sumX1Y2 += x1y2
        sumX2Y1 += x2y1

    C = N * sumX2 - sumX * sumX
    D = N * sumXY - sumX * sumY
    E = N * sumX3 + N * sumX1Y2 - (sumX2 + sumY2) * sumX
    G = N * sumY2 - sumY * sumY
    H = N * sumX2Y1 + N * sumY3 - (sumX2 + sumY2) * sumY

    denominator = C * G - D * D
    if abs(denominator) < DBL_EPSILON:
        return False

    a = (H * D - E * G) / denominator
    denominator = D * D - G * C
    if abs(denominator) < DBL_EPSILON:
        return False

    b = (H * C - E * D) / denominator
    c = -(a * sumX + b * sumY + sumX2 + sumY2) / N

    center = [0] * 2
    center[0] = int(a / (-2))
    center[1] = int(b / (-2))
    center = tuple(center)
    radius = int(np.sqrt(a * a + b * b - 4 * c) / 2)

    matchRate = 0.0
    for p in points:
        matchRate += np.linalg.norm(np.array(p) - np.array(center)) - radius

    return radius, center


def wavelet_noising(data_flow):
    """
    速度数据去噪
    """
    def sgn(num):
        if num > 0.0:
            return 1.0
        elif num == 0.0:
            return 0.0
        else:
            return -1.0

    def calc_cd(length, cd, lam, a0):
        for k in range(length):
            if abs(cd[k]) >= lam:
                cd[k] = sgn(cd[k]) * (abs(cd[k]) - a0 * lam)
            else:
                cd[k] = 0.0
        return cd

    data = data_flow

    w = pywt.Wavelet('sym8')
    # [ca1, cd1] = pywt.wavedec(data, w, level=1)  # 分解波
    # [ca3, cd3, cd2, cd1] = pywt.wavedec(data, w, level=3)  # 分解波
    [ca5, cd5, cd4, cd3, cd2, cd1] = pywt.wavedec(data, w, level=5)  # 分解波

    length0 = len(data)
    abs_cd1 = np.abs(np.array(cd1))
    median_cd1 = np.median(abs_cd1)

    sigma = (1.0 / 0.6745) * median_cd1
    lamda = sigma * np.sqrt(2.0 * np.log(float(length0)))
    usecoeffs = [ca5]

    # 软硬阈值折中的方法
    a = 0.5

    cd1 = calc_cd(len(cd1), cd1, lamda, a)
    cd2 = calc_cd(len(cd2), cd2, lamda, a)
    cd3 = calc_cd(len(cd3), cd3, lamda, a)
    cd4 = calc_cd(len(cd4), cd4, lamda, a)
    cd5 = calc_cd(len(cd5), cd5, lamda, a)

    usecoeffs.append(cd5)
    usecoeffs.append(cd4)
    usecoeffs.append(cd3)
    usecoeffs.append(cd2)
    usecoeffs.append(cd1)
    recoeffs = pywt.waverec(usecoeffs, w)
    return recoeffs


def spd_func(var, a, b, c, d):
    var = np.array(var)
    return 7.5 * a * np.sin(b * 1.884 * var + c) + 12.46 * d


def fit_curve(x, y, _debug=0):
    """
    拟合速度的正弦曲线
    速度单位：RPM
    """

    def add_x(x_orig):
        x_new = np.zeros(len(x_orig), dtype=float)
        for i in range(len(x_orig)):
            x_new[i] = sum(x_orig[:i])
        return x_new

    def smooth_y(y_orig):
        for i in range(1, len(y_orig) - 1):
            if abs(y_orig[i - 1] - y_orig[i]) >= 3 and abs(y_orig[i] - y_orig[i + 1]) >= 3:
                y_orig[i] = (y_orig[i - 1] + y_orig[i + 1]) / 2
        return y_orig

    x_1 = add_x(np.array(x))
    y_1 = smooth_y(y)
    y_1 = wavelet_noising(y_1)
    fit_a, fit_b = optimize.curve_fit(spd_func, x_1, y_1, [1, 1, 1, 1], maxfev=500000)

    if _debug:
        plt.figure(figsize=(30, 10), dpi=100)
        plt.plot(x_1, y, 'ro-', color='red', alpha=0.8, linewidth=1)
        plt.plot(x_1, y_1, 'ro-', color='blue', alpha=0.8, linewidth=1)
        plt.plot(x_1, spd_func(x_1, fit_a[0], fit_a[1], fit_a[2], fit_a[3]), color='green', alpha=0.8, linewidth=3)
        plt.show()

    return fit_a, fit_b
