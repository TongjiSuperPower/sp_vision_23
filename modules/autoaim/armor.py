import cv2
import math
import numpy as np
from modules.autoaim.transformation import LazyTransformation


class Lightbar:
    def __init__(self, h: float, angle: float, center: tuple[float, float], color: str, ratio: float) -> None:
        self.h = h  # 灯条长度
        self.angle = angle  # 与水平线夹角，顺时针增大，单位degree
        self.center = np.float32(center)
        self.color = color  # 'blue' or 'red'
        self.ratio = ratio

        # 获得沿长度方向的方向向量，指向灯条的上方
        rad = math.radians(angle)
        self.h_vector = np.float32([-math.cos(rad), -math.sin(rad)])

        self.top = self.center + 0.5 * self.h * self.h_vector
        self.bottom = self.center - 0.5 * self.h * self.h_vector
        self.points = np.float32([self.top, self.bottom])


class LightbarPair:
    def __init__(self, left: Lightbar, right: Lightbar, side_ratio: float, angle: float, ratio: float) -> None:
        self.left = left
        self.right = right
        self.side_ratio = side_ratio
        self.angle = angle
        self.ratio = ratio

        self.center = (left.center + right.center) / 2
        self.points = np.float32([left.top, right.top, right.bottom, left.bottom])


class Armor(LazyTransformation):
    def __init__(self, pair: LightbarPair, confidence: float, name: str, pattern: cv2.Mat) -> None:
        super().__init__()

        self.confidence = confidence
        self.name = name
        self.pattern = pattern

        self.left = pair.left
        self.right = pair.right
        self.center = pair.center
        self.points = pair.points
        self.color = self.left.color
