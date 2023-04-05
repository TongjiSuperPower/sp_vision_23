import cv2
import time
from remote_visualizer import Visualizer

if __name__ == '__main__':
    video_path = 'assets/input.avi'
    cap = cv2.VideoCapture(video_path)

    with Visualizer() as v:

        while True:
            success, img = cap.read()
            if not success:
                break

            v.show(img)
            cv2.waitKey(10)

        cap.release()