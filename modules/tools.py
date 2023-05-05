import cv2
import math
import numpy as np
from queue import Empty
from multiprocessing import Queue


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

    cv2.line(img, origin, imgPoints[2][0], (0, 0, 255), thickness)
    cv2.line(img, origin, imgPoints[0][0], (255, 0, 0), thickness)
    cv2.line(img, origin, imgPoints[1][0], (0, 255, 0), thickness)


def putText(img: cv2.Mat, text: str, point, color=(0, 0, 255), thickness=2) -> None:
    anchor = np.int32(point)
    cv2.putText(img, text, anchor, cv2.FONT_HERSHEY_SIMPLEX, 1, color, thickness)


def getParaTime(pos, bulletSpeed):
    '''
    用抛物线求子弹到目标位置的时间.
    pos:目标的坐标(mm);
    bulletSpeed:子弹速度(m/s);
    return: (ms).
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


def shoot_pitch(x, y, z, bullet_speed) -> float:
    g = 9.794 / 1000
    distance = (x**2 + z**2)**0.5

    a = 0.5 * g * distance**2 / bullet_speed**2
    b = -distance
    c = a - y

    result1 = (-b + math.sqrt(b**2-4*a*c))/(2*a)
    result2 = (-b - math.sqrt(b**2-4*a*c))/(2*a)
    pitch1 = math.atan(result1)
    pitch2 = math.atan(result2)
    t1 = distance / (bullet_speed * math.cos(pitch1))
    t2 = distance / (bullet_speed * math.cos(pitch2))

    pitch = pitch1 if t1 < t2 else pitch2
    pitch = math.degrees(pitch)

    return pitch


def R_gimbal2imu(yaw: float, pitch: float) -> np.ndarray:
    yaw, pitch = math.radians(yaw), math.radians(pitch)
    R_y = np.array([[math.cos(yaw), 0, math.sin(yaw)],
                    [0, 1, 0],
                    [-math.sin(yaw), 0, math.cos(yaw)]])
    R_x = np.array([[1, 0, 0],
                    [0, math.cos(pitch), -math.sin(pitch)],
                    [0, math.sin(pitch), math.cos(pitch)]])
    return R_y @ R_x


def clear_queue(q: Queue) -> None:
    try:
        while True:
            q.get(timeout=0.1)
    except Empty:
        return
    
def normalize_angle_positive(angle):
    """ Normalizes the angle to be 0 to 2*pi
        It takes and returns radians. """
    return math.fmod(math.fmod(angle, 2.0*math.pi) + 2.0*math.pi, 2.0*math.pi)

def normalize_angle(angle):
    """ Normalizes the angle to be -pi to +pi
        It takes and returns radians."""
    a = normalize_angle_positive(angle)
    if a > math.pi:
        a -= 2.0 * math.pi
    return a

def shortest_angular_distance(from_angle, to_angle):
    """ Given 2 angles, this returns the shortest angular
        difference.  The inputs and ouputs are of course radians.

        The result would always be -pi <= result <= pi. Adding the result
        to "from" will always get you an equivalent angle to "to".
    """
    return normalize_angle(to_angle - from_angle)

def is_triangle(a, b, c):
    """
    Args:
        xo 自己车中心
        xa 敌方装甲板中心
        xc 敌方车中心
        a (xo 2 xa): 
        b (xo 2 xc): 
        c (xa 2 xc): 
    """
    if a + b > c and a + c > b and b + c > a:
        return True
    else:
        return False

def triangle_angles(a, b, c):
    # 使用余弦定理计算角度
    angle_B = math.degrees(math.acos((a**2 + c**2 - b**2) / (2 * a * c)))
    return (180 - angle_B)
