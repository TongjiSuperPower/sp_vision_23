import cv2
from modules.io.recording import Recorder

if __name__ == '__main__':

    video_path = 'assets/antitop_top.mp4'

    cap = cv2.VideoCapture(video_path)

    with Recorder() as recorder:
        while True:
            success, img = cap.read()
            if not success:
                break
            
            recorder.record(img, (1,2,3))

            cv2.waitKey(30)

