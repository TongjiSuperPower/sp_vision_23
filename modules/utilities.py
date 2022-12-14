import cv2
import numpy as np


def drawContour(img: cv2.Mat, points, color=(0, 0, 255), thickness=3) -> None:
    points = np.array(points, int)
    cv2.drawContours(img, [points], -1, color, thickness)


def drawPoint(img: cv2.Mat, point, color=(0, 0, 255)) -> None:
    center = np.array(point, int)
    cv2.circle(img, center, 3, color, cv2.FILLED)


def drawAxis(img, origin, rvec, tvec, camMat, dist, scale=30, thickness=3) -> None:
    axisPoints = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.array(origin, int)

    imgPoints, _ = cv2.projectPoints(axisPoints, rvec, tvec, camMat, dist)
    imgPoints = np.array(imgPoints, int)

    cv2.line(img, origin, imgPoints[0][0], (255, 0, 0), thickness)
    cv2.line(img, origin, imgPoints[1][0], (0, 255, 0), thickness)
    cv2.line(img, origin, imgPoints[2][0], (0, 0, 255), thickness)


def putText(img: cv2.Mat, text: str, point, color=(0, 0, 255), thickness=2) -> None:
    anchor = np.array(point, int)
    cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)
