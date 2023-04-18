#!/usr/bin/env python
# coding:utf-8
import cv2
import numpy as np
import collections

from configs.NahsorConfig import *
import pywt

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


def angle_between_points(center, point1, point2):
    """
    计算point1,2与center形成的夹角角度，返回值单位为弧度(rad)
    center, point1, point2: 三个点的坐标，格式为 (x, y)
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
            _dict[COLOR.RED.name + '_2'] = [lower, upper]
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


def speed_func(t, a, w, b, t0):
    t = np.array(t)
    # return 7.5 * a * np.sin(b * 1.884 * var + c) + 12.46 * d
    # return a * np.sin(w * t + b) + 2.090 - a
    return a * np.sin(w * (t - t0)) + b


def angle_func(t, a, w, b, t0, c=0.0):
    t = np.array(t)
    return -a / w * np.cos(w * (t - t0)) + b * t + c
    # c=-a/w*cos(w*t)


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
