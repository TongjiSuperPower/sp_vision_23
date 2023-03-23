import cv2
import math
import numpy as np
from modules.classification import Classifier

# 预处理
threshold_value = 100  # 二值化阈值

# Lightbar
min_contour_area = 10  # 保证6m以内灯条面积大于该值
max_lightbar_angle = 45  # 灯条与竖直线最大夹角
min_lightbar_ratio = 2  # 最小灯条长宽比

# lightbarPair
max_angle = 45  # 装甲板左右灯条中点连线与水平线最大夹角
max_side_ratio = 2  # 装甲板左右灯条长度最大比值，max/min
min_ratio = 1  # 最小装甲板长宽比
max_ratio = 6  # 最大装甲板长宽比

# Armor
pattern_h_coefficient = 1.0  # 获得装甲板图案的上下边界的系数
margin = 50  # 透视变换后获得的图像宽度为 pattern_w + 2*margin
pattern_h, pattern_w = 100, 100  # 裁剪后所获得图案图片的大小
min_confidence = 0.8  # 判断为装甲板的最低置信度
lightbar_length, small_width, big_width = 56, 135, 230  # 单位mm


class Lightbar:
    def __init__(self, h: float, angle: float, center: tuple[float, float], color: str, area: float, ratio: float) -> None:
        self.h = h  # 灯条长度
        self.angle = angle  # 与水平线夹角，顺时针增大，单位degree
        self.center = np.float32(center)
        self.color = color  # 'blue' or 'red'
        self.area = area
        self.ratio = ratio

        # 获得沿长度方向的方向向量，指向灯条的上方
        rad = math.radians(angle)
        self.h_vector = np.float32([-math.cos(rad), -math.sin(rad)])

        self.top = self.center + 0.5 * self.h * self.h_vector
        self.bottom = self.center - 0.5 * self.h * self.h_vector
        self.points = np.float32([self.top, self.bottom])

    @property
    def passed(self) -> bool:
        area_check = self.area > min_contour_area
        angle_check = abs(self.angle-90) < max_lightbar_angle
        ratio_check = self.ratio > min_lightbar_ratio
        return area_check and angle_check and ratio_check


class LightbarPair:
    def __init__(self, left: Lightbar, right: Lightbar, side_ratio: float, angle: float, ratio: float) -> None:
        self.left = left
        self.right = right
        self.side_ratio = side_ratio
        self.angle = angle
        self.ratio = ratio

        self.center = (left.center + right.center) / 2
        self.points = np.float32([left.top, right.top, right.bottom, left.bottom])

    @property
    def passed(self) -> bool:
        side_ratio_check = self.side_ratio < max_side_ratio
        angle_check = self.angle < max_angle
        ratio_check = min_ratio < self.ratio < max_ratio
        return side_ratio_check and angle_check and ratio_check


class Armor:
    def __init__(self, pair: LightbarPair, confidence: float, name: str, pattern: cv2.Mat) -> None:
        self.confidence = confidence
        self.name = name
        self.pattern = pattern

        self.left = pair.left
        self.right = pair.right
        self.center = pair.center
        self.points = pair.points
        self.color = self.left.color

        self.abandoned = False

        self.rvec: np.ndarray = None
        self.tvec: np.ndarray = None
        self.in_camera: np.ndarray = None  # [[x], [y], [z]]
        self.in_gimbal: np.ndarray = None  # [[x], [y], [z]]
        self.in_imu: np.ndarray = None  # [[x], [y], [z]]
        self.observation: tuple[float, float, float] = None  # (z, alpha, beta)

    @property
    def passed(self) -> bool:
        confidence_check = self.confidence > min_confidence
        name_check = self.name != 'no_pattern'
        return confidence_check and name_check

    def _solve(self, cameraMatrix: np.ndarray, distCoeffs: np.ndarray, cameraVector: np.ndarray, yaw: float, pitch: float) -> None:
        # 获得装甲板中心点在相机坐标系下的坐标
        width = big_width if 'big' in self.name else small_width
        points_2d = self.points
        points_3d = np.float32([[-width / 2, -lightbar_length / 2, 0],
                                [width / 2, -lightbar_length / 2, 0],
                                [width / 2, lightbar_length / 2, 0],
                                [-width / 2, lightbar_length / 2, 0]])
        _, self.rvec, self.tvec = cv2.solvePnP(points_3d, points_2d, cameraMatrix, distCoeffs, flags=cv2.SOLVEPNP_IPPE)
        self.in_camera = self.tvec  # points_3d是以装甲板中心点为原点, 所以tvec即为装甲板中心点在相机坐标系下的坐标

        # 获得装甲板中心点在云台坐标系下的坐标
        self.in_gimbal = self.in_camera + cameraVector

        # 获得装甲板中心点在陀螺仪坐标系下的坐标
        yaw, pitch = math.radians(yaw), math.radians(pitch)
        yRotationMatrix = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                                    [0, 1, 0],
                                    [-math.sin(yaw), 0, math.cos(yaw)]])
        xRotationMatrix = np.array([[1, 0, 0],
                                    [0, math.cos(pitch), -math.sin(pitch)],
                                    [0, math.sin(pitch), math.cos(pitch)]])
        self.in_imu = yRotationMatrix @ xRotationMatrix @ self.in_gimbal

        # 将相机坐标系下的坐标转换为观测量
        x, y, z = self.in_camera.T[0]
        alpha = math.atan(x/z)  # rad
        beta = math.atan(y/z)  # rad
        self.observation = (z, alpha, beta)


class ArmorDetector:
    def __init__(self, cameraMatrix: np.ndarray, distCoeffs: np.ndarray, cameraVector: np.ndarray, classifier: Classifier) -> None:
        self._cameraMatrix = cameraMatrix
        self._distCoeffs = distCoeffs
        self._cameraVector = cameraVector
        self._classifier = classifier

        # 以下list均为未经filter的原始数据，方便调试
        self._processed_img: cv2.Mat = None
        self._lightbars: list[Lightbar] = None
        self._lightbar_pairs: list[LightbarPair] = None
        self._armors: list[Armor] = None

    def _set_processed_img(self, img: cv2.Mat) -> None:
        gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        _, threshold_img = cv2.threshold(gray_img, threshold_value, 255, cv2.THRESH_BINARY)

        self._processed_img = threshold_img

    def _set_lightbars(self, img: cv2.Mat, processed_img: cv2.Mat) -> None:
        lightbars: list[Lightbar] = []
        contours, _ = cv2.findContours(processed_img, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        for contour in contours:
            area = cv2.contourArea(contour)
            rect = cv2.minAreaRect(contour)
            center = rect[0]  # (x, y)
            h, w = rect[1]
            angle = rect[2]

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
            blue_sum = np.sum(img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 0])
            red_sum = np.sum(img[roi_y:roi_y+roi_h, roi_x:roi_x+roi_w, 2])
            color = 'blue' if blue_sum > red_sum else 'red'

            lightbar = Lightbar(h, angle, center, color, area, ratio)
            lightbars.append(lightbar)

        self._lightbars = lightbars

    def _set_lightbar_pairs(self, lightbars: list[Lightbar]) -> None:
        lightbar_pairs: list[LightbarPair] = []
        lightbars.sort(key=lambda l: l.center[0])

        for i in range(len(lightbars) - 1):
            for j in range(i + 1, len(lightbars)):
                left, right = lightbars[i], lightbars[j]

                if left.color != right.color:
                    continue

                dx, dy = np.abs(right.center - left.center)
                w = (dx ** 2 + dy ** 2) ** 0.5

                # 获得左右灯条长度比值、装甲板左右灯条中点连线与水平线夹角、装甲板长宽比
                side_ratio = max(left.h, right.h) / min(left.h, right.h)
                angle = math.atan2(dy, dx) * 180 / math.pi
                ratio = w / max(left.h, right.h)

                lightbar_pair = LightbarPair(left, right, side_ratio, angle, ratio)
                lightbar_pairs.append(lightbar_pair)

        self._lightbar_pairs = lightbar_pairs

    def _set_armors(self, img: cv2.Mat, lightbar_pairs: list[LightbarPair]) -> None:
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

        self._armors = armors

    def detect(self, img: cv2.Mat, yaw: float, pitch: float) -> list[Armor]:
        self._set_processed_img(img)

        processed_img = self._processed_img
        self._set_lightbars(img, processed_img)

        lightbars = list(filter(lambda l: l.passed, self._lightbars))
        self._set_lightbar_pairs(lightbars)

        lightbar_pairs = list(filter(lambda lp: lp.passed, self._lightbar_pairs))
        self._set_armors(img, lightbar_pairs)

        armors = list(filter(lambda a: a.passed, self._armors))

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
        armors = list(filter(lambda a: not a.abandoned, armors))

        # 对装甲板进行位置解算
        for a in armors:
            a._solve(self._cameraMatrix, self._distCoeffs, self._cameraVector, yaw, pitch)

        return armors
