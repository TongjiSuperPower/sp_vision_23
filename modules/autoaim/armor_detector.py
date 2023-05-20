import cv2
import math
import numpy as np
from itertools import combinations
from collections.abc import Iterable

from modules.autoaim.armor import Lightbar, LightbarPair, Armor
from modules.autoaim.classifier import Classifier


# 预处理
threshold_value = 100  # 二值化阈值

# Lightbar
min_contour_area = 10  # 保证6m以内灯条面积大于该值
max_lightbar_angle = 45  # 灯条与竖直线最大夹角
min_lightbar_ratio = 2  # 最小灯条长宽比
min_color_difference = 50  # 最小颜色差

# LightbarPair
max_angle = 10  # 装甲板左右灯条中点连线与水平线最大夹角
max_side_ratio = 2  # 装甲板左右灯条长度最大比值，max/min
min_ratio = 1  # 最小装甲板长宽比
max_ratio = 5  # 最大装甲板长宽比

# Armor
pattern_h_coefficient = 0.9  # 获得装甲板图案的上下边界的系数
margin = 50  # 透视变换后获得的图像宽度为 pattern_w + 2*margin
pattern_h, pattern_w = 100, 100  # 裁剪后所获得图案图片的大小
min_confidence = 0.8  # 判断为装甲板的最低置信度


def is_lightbar(l: Lightbar) -> bool:
    area_check = l.area > min_contour_area
    angle_check = abs(l.angle-90) < max_lightbar_angle
    ratio_check = l.ratio > min_lightbar_ratio
    return area_check and angle_check and ratio_check


def is_lightbar_pair(lp: LightbarPair) -> bool:
    side_ratio_check = lp.side_ratio < max_side_ratio
    angle_check = lp.angle < max_angle
    ratio_check = min_ratio < lp.ratio < max_ratio
    return side_ratio_check and angle_check and ratio_check


def is_armor(a: Armor) -> bool:
    confidence_check = a.confidence > min_confidence
    name_check = (a.name != 'no_pattern')
    return confidence_check and name_check


class ArmorDetector:
    def __init__(self, enemy_color: str) -> None:
        self._enemy_color = enemy_color
        self._classifier = Classifier()

        # 方便调试查看结果
        self._processed_img: cv2.Mat = None
        self._raw_lightbars: list[Lightbar] = None
        self._raw_lightbar_pairs: list[LightbarPair] = None
        self._raw_armors: list[Armor] = None

    def _get_processed_img(self, img: cv2.Mat) -> cv2.Mat:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

        return threshold_img

    def _get_raw_lightbars(self, img: cv2.Mat, processed_img: cv2.Mat) -> list[Lightbar]:
        lightbars: list[Lightbar] = []
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            rect = cv2.minAreaRect(contour)
            center = rect[0]  # (x, y)
            h, w = rect[1]
            angle = rect[2]
            area = h * w

            # 调整宽高，获得比例
            if h < w:
                h, w = w, h
                angle += 90
            try:
                ratio = h / w
            except ZeroDivisionError:
                continue

            # 判断颜色
            roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(contour)  # (左上x, 左上y, w, h)
            roi_blue = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 0]
            roi_red = img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 2]
            blue_sum = np.count_nonzero(cv2.subtract(roi_blue, roi_red) > min_color_difference)
            red_sum = np.count_nonzero(cv2.subtract(roi_red, roi_blue) > min_color_difference)
            color = 'blue' if blue_sum > red_sum else 'red'
            if color != self._enemy_color:
                continue

            lightbar = Lightbar(h, angle, center, color, area, ratio)
            lightbars.append(lightbar)

        return lightbars

    def _get_raw_lightbar_pairs(self, lightbars: Iterable[Lightbar]) -> list[LightbarPair]:
        lightbar_pairs: list[LightbarPair] = []
        lightbars = sorted(lightbars, key=lambda l: l.center[0])

        for left, right in combinations(lightbars, 2):
            dx, dy = np.abs(right.center - left.center)
            w = (dx ** 2 + dy ** 2) ** 0.5

            # 获得左右灯条长度比值、装甲板左右灯条中点连线与水平线夹角、装甲板长宽比
            side_ratio = max(left.h, right.h) / min(left.h, right.h)
            angle = math.atan2(dy, dx) * 180 / math.pi
            ratio = w / max(left.h, right.h)

            lightbar_pair = LightbarPair(left, right, side_ratio, angle, ratio)
            lightbar_pairs.append(lightbar_pair)

        return lightbar_pairs

    def _get_raw_armors(self, img: cv2.Mat, lightbar_pairs: Iterable[LightbarPair]) -> list[Armor]:
        armors: list[Armor] = []

        for lightbar_pair in lightbar_pairs:
            left, right = lightbar_pair.left, lightbar_pair.right

            # 获得装甲板图案四个角点
            top_left = left.center + left.h * left.h_vector * pattern_h_coefficient
            bottom_left = left.center - left.h * left.h_vector * pattern_h_coefficient
            top_right = right.center + right.h * right.h_vector * pattern_h_coefficient
            bottom_right = right.center - right.h * right.h_vector * pattern_h_coefficient
            from_points = np.float32([top_left, top_right, bottom_right, bottom_left])

            # 透视变换获得图案图片
            h, w = pattern_h, pattern_w + margin * 2
            to_points = np.float32(((0, 0), (w, 0), (w, h), (0, h)))
            transform = cv2.getPerspectiveTransform(from_points, to_points)
            pattern = cv2.warpPerspective(img, transform, (w, h))

            # 裁剪两侧灯条
            pattern = pattern[:, margin:-margin]

            # 用高斯模糊+大津法提取图案
            pattern = cv2.cvtColor(pattern, cv2.COLOR_BGR2GRAY)
            pattern = cv2.GaussianBlur(pattern, (5, 5), 0)
            _, pattern = cv2.threshold(pattern, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

            # 分类器分类
            confidence, name = self._classifier.classify(pattern)

            armor = Armor(lightbar_pair, confidence, name, pattern)
            armors.append(armor)

        return armors

    def detect(self, img: cv2.Mat) -> Iterable[Armor]:
        self._processed_img = self._get_processed_img(img)

        self._raw_lightbars = self._get_raw_lightbars(img, self._processed_img)
        lightbars = filter(lambda l: is_lightbar(l), self._raw_lightbars)

        self._raw_lightbar_pairs = self._get_raw_lightbar_pairs(lightbars)
        lightbar_pairs = filter(lambda lp: is_lightbar_pair(lp), self._raw_lightbar_pairs)

        self._raw_armors = self._get_raw_armors(img, lightbar_pairs)
        armors = filter(lambda a: is_armor(a), self._raw_armors)

        return armors
