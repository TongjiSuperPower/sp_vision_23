import cv2
import math
import numpy as np


class LightBar:
    def __init__(self, len: float, angle: float, center) -> None:
        self.len = len
        self.angle = angle  # 与水平线夹角，顺时针增大
        self.center = np.float32(center)

        rad = angle / 180 * math.pi
        self.vector = np.float32([-math.cos(rad), -math.sin(rad)])  # 指向灯条的上方

        top = self.center + 0.5 * self.len * self.vector
        bottom = self.center - 0.5 * self.len * self.vector
        self.points = np.float32([top, bottom])

    def __str__(self) -> str:
        return f'LightBar({self.len}, {self.angle}, {self.center})'

    def __repr__(self) -> str:
        return self.__str__()


class Armor:
    def __init__(self, left: LightBar, right: LightBar) -> None:
        topLeft = left.center + left.len * left.vector
        bottomLeft = left.center - left.len * left.vector
        topRight = right.center + right.len * right.vector
        bottomRight = right.center - right.len * right.vector
        self.points = np.float32([topLeft, topRight, bottomRight, bottomLeft])
        self.imgPoints = np.float32([left.points[0], right.points[0], right.points[1], left.points[1]])

        # after self.tagered(...)
        self.rvec = None
        self.tvec = None
        self.center = None
        self.aimPoint = None
        self.yaw = None
        self.pitch = None

    def targeted(self, objPoints, cameraMatrix, distCoeffs):
        _, self.rvec, self.tvec = cv2.solvePnP(objPoints, self.imgPoints, cameraMatrix, distCoeffs)
        self.center = cv2.projectPoints(np.float32([[0, 0, 0]]), self.rvec, self.tvec, cameraMatrix, distCoeffs)[0][0][0]

        rotationMatrix, _ = cv2.Rodrigues(self.rvec)
        self.aimPoint = (np.dot(rotationMatrix, np.float32([0, 0, 0]).reshape(3, 1)) + self.tvec).reshape(1, 3)[0]

        x, y, z = self.aimPoint  # 相机坐标系：相机朝向为z正方向，相机右侧为x正方向，相机下侧为y正方向，符合右手系

        yaw = cv2.fastAtan2(x, (y**2 + z**2)**0.5)
        self.yaw = yaw - 360 if yaw > 180 else yaw

        pitch = cv2.fastAtan2(y, (x**2 + z**2)**0.5)
        self.pitch = pitch - 360 if pitch > 180 else pitch

    def __str__(self) -> str:
        return f'Armor({self.points})'

    def __repr__(self) -> str:
        return self.__str__()


class Detector:
    def __init__(self) -> None:
        # TODO ROI
        pass

    def detect(self, img: cv2.Mat) -> tuple[tuple[LightBar], tuple[Armor]]:
        blue, _, red = cv2.split(img)
        subtracted = cv2.subtract(blue, red)
        _, threshed = cv2.threshold(subtracted, 150, 255, cv2.THRESH_BINARY)
        contours, _ = cv2.findContours(threshed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        lightBars = []
        for contour in contours:
            if cv2.contourArea(contour) > 50:
                rect = cv2.minAreaRect(contour)
                center = rect[0]
                h, w = rect[1]
                angle = rect[2]  # 与水平线夹角，顺时针增大

                if h < w:
                    h, w = w, h
                    angle += 90
                ratio = h / w

                if h > 50 and ratio > 3.5 and 45 < angle < 135:
                    lightBars.append(LightBar(h, angle, center))

        lightBars.sort(key=lambda x: x.center[0])

        armors = []
        for i in range(len(lightBars) - 1):
            l = lightBars[i]
            r = lightBars[i + 1]
            dx, dy = r.center - l.center
            w = (dx ** 2 + dy ** 2) ** 0.5
            ratio = 2 * w / (l.len + r.len)
            centerLineAngle = math.atan2(-dy, dx) * 180 / math.pi
            heightRatio = abs(l.len - r.len) / max(l.len, r.len)

            if abs(l.angle - r.angle) < 8 and abs(centerLineAngle) < 60 and 1 < ratio < 3 and heightRatio < 0.3:
                armors.append(Armor(l, r))

        return lightBars, armors
