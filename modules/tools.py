import cv2
import math
import numpy as np


class VirtualComu():
    yaw=10.
    pitch=2.
    def __init__(self) -> None:
        pass
    def received():
        return True
    def send(a):
        print('virtualSent')


def drawContour(img: cv2.Mat, points, color=(0, 0, 255), thickness=3) -> None:
    points = np.int32(points)
    cv2.drawContours(img, [points], -1, color, thickness)


def drawPoint(img: cv2.Mat, point, color=(0, 0, 255), radius=3, thickness=None) -> None:
    center = np.int32(point)
    if thickness == None:
        cv2.circle(img, center, radius, color, cv2.FILLED)
    else:
        cv2.circle(img, center, radius, color, thickness)


def drawAxis(img, origin, rvec, tvec, cameraMatrix, distCoeffs, scale=30, thickness=3) -> None:
    '''x: blue, y: green, z: red'''
    axisPoints = scale * np.float32([[1, 0, 0], [0, 1, 0], [0, 0, 1]])
    origin = np.int32(origin)

    imgPoints, _ = cv2.projectPoints(axisPoints, rvec, tvec, cameraMatrix, distCoeffs)
    imgPoints = np.int32(imgPoints)

    cv2.line(img, origin, imgPoints[0][0], (255, 0, 0), thickness)
    cv2.line(img, origin, imgPoints[1][0], (0, 255, 0), thickness)
    cv2.line(img, origin, imgPoints[2][0], (0, 0, 255), thickness)


def putText(img: cv2.Mat, text: str, point, color=(0, 0, 255), thickness=2) -> None:
    anchor = np.int32(point)
    cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)


def getParaTime(pos, bulletSpeed):
    '''
    用抛物线求子弹到目标位置的时间.
    pos:目标的坐标(mm);
    bulletSpeed:子弹速度(m/s);
    '''
    pos = np.reshape(pos, (3,))
    x = pos[0]
    y = pos[1]
    z = pos[2]

    dxz = math.sqrt(x*x+z*z)
    a = 0.5*9.7940/1000*dxz*dxz/(bulletSpeed*bulletSpeed)
    b = dxz
    c = a - y

    res1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
    res2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)

    beta1 = math.atan(res1)
    beta2 = math.atan(res2)

    t1 = dxz/(bulletSpeed*math.cos(beta1))
    t2 = dxz/(bulletSpeed*math.cos(beta2))

    # t = math.sqrt(x**2+y**2+z**2)/bulletSpeed

    t = t1 if t1 < t2 else t2

    return t



def compensateGravity(pos, bulletSpeed):
    '''
    重力补偿。输入世界坐标(mm)和弹速(m/s)，输出补偿后的世界坐标
    '''
    flyTime = getParaTime(pos, bulletSpeed)
    
    dropDistance = 0.5 * 9.7940/1000 * flyTime**2

    pos[1] -= dropDistance # 因为y轴方向向下，所以是减法

    return pos

