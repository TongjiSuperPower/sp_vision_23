#!/usr/bin/env python
# coding:utf-8
import time

from modules.Nahsor.Nahsor import *
import cv2 as cv

def recognise():
    filename = "../assets/new_mid_1.MP4"
    cap = cv.VideoCapture(filename)

    if not cap.isOpened():
        print("Cannot open camera")
        exit()
    color = COLOR.RED
    # 新建能量机关对象
    w = NahsorMarker(color=color, fit_debug=0, target_debug=1,
                     fit_speed_mode=FIT_SPEED_MODE.CURVE_FIT)

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
            pfps = rfps
            rfps = 0

        # 逐帧捕获
        ret, img = cap.read()
        # 如果正确读取帧，ret为True
        if not ret:
            break


        # 使用mark()方法，传入一帧图像
        w.mark(img)
        # 使用markFrame()获得标记好的输出图像
        img = w.markFrame()
        # 显示帧率
        img = cv2.putText(img, 'fps:'+str(pfps), (0, 80), cv2.FONT_HERSHEY_COMPLEX, 1.5, (100, 200, 200), 5)


        cv2.namedWindow("Press q to end", cv2.WINDOW_NORMAL)
        cv2.imshow("Press q to end", img)
        # cv2.waitKey()
    cap.release()
    cv.destroyAllWindows()


if __name__ == "__main__":
    recognise()
