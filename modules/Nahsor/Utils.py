#!/usr/bin/env python
# coding:utf-8
import cv2
import numpy as np
import collections

from scipy.interpolate import interp1d
from scipy.ndimage import gaussian_filter1d
from scipy.optimize import leastsq

from NahsorConfig import *
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


def get_distance(p1, p2):
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


def angle_between_points(center, point1, point2):
    """
    计算point1,2与center形成的夹角角度，返回值单位为弧度(rad)
    A, B, C: 三个点的坐标，格式为 (x, y)
    """
    AB = np.array(point1) - np.array(center)
    AC = np.array(point2) - np.array(center)
    dot_product = np.dot(AB, AC)
    cos_angle = dot_product / (np.linalg.norm(AB) * np.linalg.norm(AC))
    if cos_angle > 1:
        cos_angle = 1
    elif cos_angle < -1:
        cos_angle = -1
    angle = np.arccos(cos_angle)
    return angle


# def get_distance(width, w_scale):
#     """
#     返回所在点与装甲板目标的距离，单位：m
#     """
#     dis = ((TARGET_WIDTH / w_scale) * FOCAL_LENGTH) / (width * 100)
#     return "%.2fm" % dis


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


def get_clockwise(center, last_center, current_center):
    """
    返回大符的转动方向
    """
    last_center = np.array(last_center)
    current_center = np.array(current_center)
    center = np.array(center)
    # 计算向量
    vec1 = last_center - center
    vec2 = current_center - center

    # 计算向量夹角
    angle = np.arccos(np.dot(vec1, vec2) / (np.linalg.norm(vec1) * np.linalg.norm(vec2))) * 180 / np.pi

    # 计算向量叉积
    cross_product = np.cross(vec1, vec2)

    # 判断方向
    if angle < 180 and cross_product > 0:
        return 1  # 顺时针
    else:
        return -1  # 逆时针


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


def get_color_range(color, color_space):
    """
    定义字典存放颜色分量上下限
    例如：{颜色: [min分量, max分量]}
    'red': [array([160,  43,  46]), array([179, 255, 255])]}
    """
    _dict = collections.defaultdict(list)
    if color_space == COLOR_SPACE.HSV:
        if color == COLOR.RED:
            # HSV红色
            lower = np.array(HSV_RED_LOWER_1)
            upper = np.array(HSV_RED_UPPER_1)
            _dict[COLOR.RED] = [lower, upper]
            # HSV红色2
            lower = np.array(HSV_RED_LOWER_2)
            upper = np.array(HSV_RED_UPPER_2)
            _dict[COLOR.RED.name+'_2'] = [lower, upper]
        else:
            # HSV蓝色
            lower = np.array(HSV_BLUE_LOWER)
            upper = np.array(HSV_BLUE_UPPER)
            _dict[COLOR.BLUE] = [lower, upper]
    elif color_space == COLOR_SPACE.YUV:
        if color == COLOR.RED:
            # YUV红色
            lower = np.array(YUV_RED_LOWER)
            upper = np.array(YUV_RED_UPPER)
            _dict[COLOR.RED] = [lower, upper]
        else:
            # YUV蓝色
            lower = np.array(YUV_BLUE_LOWER)
            upper = np.array(YUV_BLUE_UPPER)
            _dict[COLOR.BLUE] = [lower, upper]
    elif color_space == COLOR_SPACE.BGR:
        if color == COLOR.RED:
            # BGR红色
            lower = np.array(BGR_RED_LOWER)
            upper = np.array(BGR_RED_UPPER)
            _dict[COLOR.RED] = [lower, upper]
        else:
            # BGR蓝色
            lower = np.array(BGR_BLUE_LOWER)
            upper = np.array(BGR_BLUE_UPPER)
            _dict[COLOR.BLUE] = [lower, upper]
    elif color_space == COLOR_SPACE.SIMU_BLACK:
        # 模拟+黑化 红色
        lower = np.array(SIMU_BLACK_RED_LOWER)
        upper = np.array(SIMU_BLACK_RED_UPPER)
        _dict[COLOR.RED] = [lower, upper]
    else:
        # HSV红色
        lower = np.array(HSV_RED_LOWER_1)
        upper = np.array(HSV_RED_UPPER_1)
        _dict[COLOR.RED] = [lower, upper]

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
        if 0.65 <= (get_distance(tmp_r_center, p_center) / pLen) <= 1.3:
            # 限制R与父轮廓中心的连线角度
            sim = abs(calc_point_angle(p_center, tmp_r_center) - rect_armor[-1])
            if sim <= MAX_R_ANGLE or abs(sim - 90) or abs(sim - 180) <= MAX_R_ANGLE:
                r_center = (int(tmp_r_center[0]), int(tmp_r_center[1]))
                radius = int(get_distance(r_center, cur_point))
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
        if -2 <= get_distance(m, center) - w / 2 <= 2:
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


# def wavelet_noising(data_flow):
#     """
#     速度数据去噪
#     """
#
#     def sgn(num):
#         if num > 0.0:
#             return 1.0
#         elif num == 0.0:
#             return 0.0
#         else:
#             return -1.0
#
#     def calc_cd(length, cd, lam, a0):
#         for k in range(length):
#             if abs(cd[k]) >= lam:
#                 cd[k] = sgn(cd[k]) * (abs(cd[k]) - a0 * lam)
#             else:
#                 cd[k] = 0.0
#         return cd
#
#     data = data_flow
#
#     w = pywt.Wavelet('sym8')
#     w = pywt.Wavelet('db4')
#     # 填充输入数据，使其长度为2的整数次幂
#     padded_data = pywt.pad(data, 2 ** int(np.ceil(np.log2(len(data)))), 'symmetric')
#
#     # 计算分解小波层数，使得分解后的系数长度等于输入数据长度
#     level = int(np.floor(np.log2(len(padded_data))) - np.log2(len(w.dec_hi)))
#
#     # 进行小波分解
#     coeffs = pywt.wavedec(padded_data, w, level)
#
#     # 计算噪声阈值
#     abs_coeffs = np.abs(np.concatenate(coeffs[1:]))
#     median_coeffs = np.median(abs_coeffs)
#     sigma = (1.0 / 0.6745) * median_coeffs
#     lamda = sigma * np.sqrt(2.0 * np.log(float(len(data))))
#
#     # 软硬阈值折中的方法
#     a = 0.5
#
#     # 进行系数去噪
#     for i in range(1, len(coeffs)):
#         coeffs[i] = calc_cd(len(coeffs[i]), coeffs[i], lamda, a)
#
#     # 进行小波重构，得到长度与输入数据相等的输出数据
#     output_data = pywt.waverec(coeffs, w)
#     output_data = output_data[:len(data)]
#
#     return output_data


def speed_func(t, a, w, t0, b):
    t = np.array(t)
    # return 7.5 * a * np.sin(b * 1.884 * var + c) + 12.46 * d
    # return a * np.sin(w * t + b) + 2.090 - a
    return a * np.sin(w * (t - t0)) + b


def angle_func(t, a, w, t0, b, c):
    t = np.array(t)
    return -a / w * np.cos(w * (t - t0)) + b * t + c
    # c=-a/w*cos(w*t)


def add_list(old_list):
    new_list = np.zeros(len(old_list), dtype=float)
    for i in range(len(old_list)):
        new_list[i] = sum(old_list[0:i + 1])
    return new_list


def get_r_by_contours(contours, parent_contours, target_center, target_radius):
        max_area = float('-inf')
        r_contour = None
        for parent_contour_number, child_contours in parent_contours.items():
            num_sub_contours = len(child_contours)
            # 没有子轮廓
            if num_sub_contours == 0:
                parent_contour = contours[parent_contour_number]
                parent_rect = cv2.minAreaRect(parent_contour)
                parent_width = max(parent_rect[1][0], parent_rect[1][1])
                parent_height = min(parent_rect[1][0], parent_rect[1][1])
                # 方形且距离在一定范围
                if SQUARE_WH_RATIO[0] < parent_width / parent_height < SQUARE_WH_RATIO[1] and CENTER_DISTANCE_RATIO[
                    0] < get_distance(target_center, parent_rect[0]) / target_radius < CENTER_DISTANCE_RATIO[1]:
                    if cv2.contourArea(parent_contour) > max_area:
                        max_area = cv2.contourArea(parent_contour)
                        r_contour = parent_contour

        if r_contour is not None:
            return cv2.minAreaRect(r_contour)[0]
        else:
            return None

def get_r_by_centers(target_centers):
        """
        通过最小二乘法来拟合圆的信息
        参数：self.cirV: 所有点坐标
        返回值: center: 得到的圆心坐标
               radius: 圆的半径
        """

        centers = target_centers
        if len(centers) < 20:
            return None

        [sumX, sumY, sumX2, sumY2, sumX3, sumY3, sumXY, sumX1Y2, sumX2Y1] = [0.0] * 9
        N = len(centers)
        for i in range(len(centers)):
            x, y = centers[i][0], centers[i][1]
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
            return None

        a = (H * D - E * G) / denominator
        denominator = D * D - G * C
        if abs(denominator) < DBL_EPSILON:
            return None

        b = (H * C - E * D) / denominator
        c = -(a * sumX + b * sumY + sumX2 + sumY2) / N

        r_center = [0] * 2
        r_center[0] = int(a / (-2))
        r_center[1] = int(b / (-2))
        r_center = tuple(r_center)
        # r_radius = int(np.sqrt(a * a + b * b - 4 * c) / 2)

        # matchRate = 0.0
        # for p in centers:
        #     matchRate += np.linalg.norm(np.array(p) - np.array(r_center)) - r_radius

        return r_center

def get_parent_contours(contours, hierarchy):
        # ----------- 按约束条件筛选轮廓 start -----------
        parent_contours = {}
        # 寻找有方形子轮廓的轮廓
        for i, contour in enumerate(contours):
            if cv2.contourArea(contour) < CONTOUR_MIN_AREA:
                continue
            if hierarchy[0][i][3] == -1:
                parent_contours[i] = []
            else:
                try:
                    parent_contours[hierarchy[0][i][3]].append(i)
                except KeyError:
                    pass

        return parent_contours

def get_target(contours, parent_contours):
        # 统计子轮廓数量，并记录面积最大的子轮廓为方型的最大的轮廓
        max_area = float('-inf')
        target_contour = None
        for parent_contour_number, child_contours in parent_contours.items():
            if len(child_contours) > 0:
                parent_contour = contours[parent_contour_number]
                child_max_area = float('-inf')
                # 找出最大的子轮廓
                for child_contour_number in child_contours:
                    child_contour = contours[child_contour_number]
                    if cv2.contourArea(child_contour) > child_max_area:
                        child_max_area = cv2.contourArea(child_contour)
                        max_child_contour = child_contour
                max_child_rect = cv2.minAreaRect(max_child_contour)
                max_child_width = max(max_child_rect[1][0], max_child_rect[1][1])
                max_child_height = min(max_child_rect[1][0], max_child_rect[1][1])
                # 找最大的子轮廓是方形的轮廓中最大的轮廓
                if SQUARE_WH_RATIO[0] < max_child_width / max_child_height < SQUARE_WH_RATIO[1] and cv2.contourArea(
                        parent_contour) > max_area:
                    max_area = cv2.contourArea(parent_contour)
                    target_contour = max_child_contour

        return target_contour