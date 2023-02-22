# 捕获用于标定的图片
import cv2
import numpy as np
import os
import sys
import toml

from modules.mindvision import Camera

useCamera = True
cap = Camera(2) if useCamera else cv2.VideoCapture('assets/input.avi')

f=1
while True:
    success, frame = cap.read()
    if not success:
        break
    cv2.imshow('ori',frame)
    res = (cv2.waitKey(30) & 0xFF)
    if res == ord('q'):
        break
    elif res == ord('c'):
        name = str(f) + ".jpg"
        cv2.imwrite(name, frame)
        f+=1
    
            

cap.release()