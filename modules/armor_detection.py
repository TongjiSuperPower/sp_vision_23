import cv2
import math
import numpy as np
from modules.classification import Classifier

threshold_value = 150  # 二值化阈值
min_contour_area = 10  # 保证6m以内灯条面积大于该值
max_lightbar_angle = 45  # 灯条与竖直线最大夹角
min_lightbar_ratio = 2  # 最小灯条长宽比

max_angle = 45  # 装甲板左右灯条中点连线与水平线最大夹角
max_side_ratio = 2  # 装甲板左右灯条长度最大比值，max/min
min_ratio = 1  # 最小装甲板长宽比
max_ratio = 6  # 最大装甲板长宽比

pattern_h_coefficient = 1.0  # 获得装甲板图案的上下边界的系数
margin = 50  # 透视变换后获得的图像宽度为 pattern_w + 2*margin
pattern_h, pattern_w = 100, 100  # 裁剪后所获得图案图片的大小
min_confidence = 0.8  # 判断为装甲板的最低置信度

lightbar_length, small_width, big_width = 56, 135, 230  # 单位mm


class Lightbar:
    def __init__(self, h: float, angle: float, center: tuple[float, float], color: str) -> None:
        self.h = h  # 灯条高度
        self.angle = angle  # 与水平线夹角，顺时针增大，单位degree
        self.center = np.float32(center)
        self.color = color  # 'blue' or 'red'

        rad = angle / 180 * math.pi
        self.h_vector = np.float32([-math.cos(rad), -math.sin(rad)])  # 高度方向向量，指向灯条的上方

        self.top = self.center + 0.5 * self.h * self.h_vector
        self.bottom = self.center - 0.5 * self.h * self.h_vector
        self.points = np.float32([self.top, self.bottom])


class Pattern:
    def __init__(self, img: cv2.Mat, name: str, confidence: float) -> None:
        self.img = img
        self.name = name
        self.confidence = confidence


class Armor:
    def __init__(self, left: Lightbar, right: Lightbar, confidence: float, class_name: str, pattern_points: np.ndarray) -> None:
        self.left = left
        self.right = right

        self.color = left.color
        self.confidence = confidence
        self.class_name = class_name
        self.points = pattern_points

        width = big_width if 'big' in class_name else small_width
        self.pnp_obj_points = np.float32([[-width / 2, -lightbar_length / 2, 0],
                                          [width / 2, -lightbar_length / 2, 0],
                                          [width / 2, lightbar_length / 2, 0],
                                          [-width / 2, lightbar_length / 2, 0]])
        self.pnp_img_points = np.float32([left.top, right.top, right.bottom, left.bottom])

        self.abandoned = False

    def in_camera(self, camera_intrinsic, camera_distortion) -> tuple[float, float, float]:
        '''获得装甲板中心点在相机坐标系下的坐标'''
        _, rvec, tvec = cv2.solvePnP(self.pnp_obj_points, self.pnp_img_points, camera_intrinsic, camera_distortion, flags=cv2.SOLVEPNP_IPPE)
        x, y, z = tvec.T[0]  # pnp_obj_points是以装甲板中心点为原点
        return x, y, z


class ArmorDetector:
    def __init__(self, classifier: Classifier) -> None:
        self.classifier = classifier

    def detect(self, img: cv2.Mat) -> tuple[tuple[Lightbar], tuple[Armor], tuple[Pattern]]:
        # 图像预处理
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

        # 筛选灯条
        lightbars: list[Lightbar] = []
        contours, _ = cv2.findContours(threshold_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for contour in contours:
            area = cv2.contourArea(contour)
            if area > min_contour_area:
                rect = cv2.minAreaRect(contour)
                center = rect[0]  # (x, y)
                h, w = rect[1]
                angle = rect[2]

                if h < w:
                    h, w = w, h
                    angle += 90
                ratio = h / w

                if abs(angle-90) < max_lightbar_angle and min_lightbar_ratio < ratio:
                    # 判断灯条颜色
                    roi_x, roi_y, roi_w, roi_h = cv2.boundingRect(contour)  # (左上x，左上y，w，h)
                    blue_sum = np.sum(img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 0])
                    red_sum = np.sum(img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 2])
                    color = 'blue' if blue_sum > red_sum else 'red'

                    points = cv2.boxPoints(rect)
                    lightbars.append(Lightbar(h, angle, center, color))

        # 筛选装甲板
        armors: list[Armor] = []
        patterns: list[Pattern] = []
        lightbars.sort(key=lambda l: l.center[0])
        for i in range(len(lightbars) - 1):
            for j in range(i + 1, len(lightbars)):
                left, right = lightbars[i], lightbars[j]

                if left.color == right.color:
                    side_ratio = max(left.h, right.h) / min(left.h, right.h)
                    dx, dy = np.abs(right.center - left.center)
                    angle = math.atan2(dy, dx) * 180 / math.pi
                    w = (dx ** 2 + dy ** 2) ** 0.5
                    ratio = w / max(left.h, right.h)

                    if side_ratio < max_side_ratio and angle < max_angle and min_ratio < ratio < max_ratio:
                        # 获得装甲板图案四个角点
                        top_left = left.center + left.h * left.h_vector * pattern_h_coefficient
                        bottom_left = left.center - left.h * left.h_vector * pattern_h_coefficient
                        top_right = right.center + right.h * right.h_vector * pattern_h_coefficient
                        bottom_right = right.center - right.h * right.h_vector * pattern_h_coefficient
                        pattern_points = np.float32([top_left, top_right, bottom_right, bottom_left])

                        # 透视变换获得图案图片
                        from_points = pattern_points
                        h, w = pattern_h, pattern_w + margin * 2
                        to_points = np.float32(((0, 0), (w, 0), (w, h), (0, h)))
                        transform = cv2.getPerspectiveTransform(from_points, to_points)
                        pattern_img = cv2.warpPerspective(img, transform, (w, h))

                        # 裁剪两侧灯条
                        pattern_img = pattern_img[:, margin:-margin]

                        # 用高斯模糊+大津法提取图案
                        pattern_img = cv2.cvtColor(pattern_img, cv2.COLOR_BGR2GRAY)
                        pattern_img = cv2.GaussianBlur(pattern_img, (5, 5), 0)
                        _, pattern_img = cv2.threshold(pattern_img, 0, 255, cv2.THRESH_BINARY+cv2.THRESH_OTSU)

                        # 分类器分类
                        confidence, class_name = self.classifier.classify(pattern_img)

                        # 调试用
                        patterns.append(Pattern(pattern_img, class_name, confidence))

                        if confidence > min_confidence and class_name != 'no_pattern':
                            armors.append(Armor(left, right, confidence, class_name, pattern_points))

        # 分类器可能会误识别，如果两个装甲板共用一个灯条，选择置信度高的
        armors.sort(key=lambda a: a.confidence, reverse=True)  # 按置信度从大到小排列
        for i in range(len(armors) - 1):
            a = armors[i]
            if a.abandoned:
                continue

            for j in range(i + 1, len(armors)):
                b = armors[j]

                if b.left in (a.left, a.right) or b.right in (a.left, a.right):
                    b.abandoned = True
        armors = filter(lambda a: not a.abandoned, armors)

        return lightbars, armors, patterns
