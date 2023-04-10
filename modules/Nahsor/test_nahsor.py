#!/usr/bin/env python
# coding:utf-8
import time

from Nahsor import *
import cv2 as cv

"""
使用测试函数，将本文件放到与Nahsor文件夹同级目录下
"""


def darkimg(img):
    img_dark = img
    # img_dark = cv2.convertScaleAbs(img_dark, alpha=1)
    img_dark = cv2.addWeighted(img_dark, 1, np.zeros(img.shape, img.dtype), 0, -150)
    img_dark = cv2.addWeighted(img_dark, 2.5, np.zeros(img.shape, img.dtype), 0, 0)
    # # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3,3))
    # # img_dark = cv2.morphologyEx(img_dark, cv2.MORPH_CLOSE, kernel)
    # img_dark = cv2.addWeighted(img_dark, 1, np.zeros(img.shape, img.dtype), 0, -230)
    # img_dark = cv2.addWeighted(img_dark, 20, np.zeros(img.shape, img.dtype), 0, -50)
    # img_dark = img
    # img_dark = cv2.convertScaleAbs(img_dark, alpha=10)
    #
    # img_dark = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, -225)
    # img_dark = cv2.addWeighted(img_dark, 7, np.zeros(img.shape, img.dtype), 0, 0)
    # img_dark = cv2.addWeighted(img_dark, 1, np.zeros(img.shape, img.dtype), 0, -40)
    # img_dark = cv2.addWeighted(img_dark, 2, np.zeros(img.shape, img.dtype), 0, 0)
    # kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
    # img_dark = cv2.morphologyEx(img_dark, cv2.MORPH_OPEN, kernel)
    # img_dark = cv2.morphologyEx(img_dark, cv2.MORPH_CLOSE, kernel)
    # upper_r = (255, 255, 255)
    # lower_r = (120, 130, 220)
    # img_dark = cv2.inRange(img_dark, lower_r, upper_r)

    return img_dark


def recognise():
    filename = "new_mid_3.MP4"
    cap = cv.VideoCapture(filename)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    color = COLOR.RED
    # 新建能量机关对象
    w = NahsorMarker(color=color, fit_debug=1,
                     fit_speed_mode=FIT_SPEED_MODE.BY_SPEED)

    # 帧率计算
    nowt = time.time()
    last = time.time()
    rfps = 0
    pfps = 0
    while (cv2.waitKey(1) & 0xFF) != ord('q'):
        rfps = rfps + 1
        nowt = time.time()
        if nowt - last >= 1:
            last = nowt
            # print(rfps)
            pfps = rfps
            rfps = 0

        # 逐帧捕获
        ret, img = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            break
        # 显示帧率

        # img=darkimg(img)
        # 使用mark()方法，传入一帧图像
        start_time = time.time()
        w.mark(img)
        print("Start", time.time() - start_time)
        # 使用markFrame()获得标记好的输出图像
        img = w.markFrame()
        img = cv2.putText(img, str(pfps), (0, 50), cv2.FONT_HERSHEY_COMPLEX, 2.0, (100, 200, 200), 5)

        # 使用getResult()方法获得输出
        # print(w.getResult())
        cv2.namedWindow("Press q to end", cv2.WINDOW_NORMAL)
        cv2.imshow("Press q to end", img)
        # time.sleep(10)
    cap.release()
    # out.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    recognise()
