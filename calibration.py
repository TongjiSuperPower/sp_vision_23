import cv2
import math
import numpy as np
from modules.mindvision import Camera
from modules.utilities import drawContour, drawPoint, drawAxis, putText


def calibrate_aperture():
    '''调节光圈使相机能看清5m远装甲板的图案'''
    cap = Camera(40)

    while True:
        success, img = cap.read()
        if not success:
            continue

        h, w, _ = img.shape
        row_s = h//4
        row_e = 3*h//4
        col_s = w//4
        col_e = 3*w//4

        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        roi = gray[row_s: row_e, col_s: col_e]
        sharpness = cv2.Laplacian(roi, cv2.CV_64F).var()

        # gray =
        gray = img
        drawContour(gray, [(col_s, row_s), (col_e, row_s), (col_e, row_e), (col_s, row_e)])
        putText(gray, f'{sharpness=:.2f}', (col_s, row_s), (255, 255, 255))
        cv2.imshow('press q to exit', gray)

        key = (cv2.waitKey(1) & 0xFF)
        if key == ord('q'):
            break

    cap.release()


def calibrate_intrinsic_and_distortion():
    '''标定相机内参和畸变系数'''

    # 该方法使用棋盘格图案标定，效果不理想
    # 建议使用圆点标定板配合matlab进行标定，效果很好

    criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

    # 标定板参数设置
    patternSize = (8, 10)
    square_length = 15  # 每个方格的大小，单位mm
    objCorners = np.zeros((patternSize[0] * patternSize[1], 3), np.float32)
    objCorners[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2) * square_length

    cap = Camera(10)

    captures = []
    while True:
        success, frame = cap.read()
        if not success:
            continue

        cv2.imshow('press c to capture, d to calibrate, q to exit', frame)

        key = (cv2.waitKey(1) & 0xFF)

        if key == ord('q'):
            break

        elif key == ord('c'):
            captures.append(frame.copy())
            print(f'captured {len(captures)}')

        elif key == ord('d'):
            object_points = []
            image_points = []
            for img in captures:
                gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                success, imgCorners = cv2.findChessboardCorners(gray, patternSize)
                if success:
                    imgCorners = cv2.cornerSubPix(gray, imgCorners, (11, 11), (-1, -1), criteria)
                    cv2.drawChessboardCorners(img, patternSize, imgCorners, success)
                    object_points.append(objCorners)
                    image_points.append(imgCorners)
                print(success)

            _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(
                object_points,
                image_points,
                gray.shape[::-1],
                None,
                None
            )
            print(f'camera_intrinsic = np.float32({mtx.tolist()})')
            print(f'camera_distortion = np.float32({dist[0].tolist()})')

            mean_error = 0
            for i in range(len(object_points)):
                projected_points, _ = cv2.projectPoints(
                    object_points[i],
                    rvecs[i],
                    tvecs[i],
                    mtx,
                    dist
                )
                error = cv2.norm(image_points[i], projected_points, cv2.NORM_L2) / len(projected_points)
                mean_error += error
            mean_error /= len(object_points)
            print(f"error: {mean_error}")
            break

    cap.release()


def calibrate_vector_to_gimbal():
    # 未实现
    # 建议使用matlab获得相机光心到标定板距离，
    # 同时测量相机背面到标定板的物理距离，
    # 求差，即为光心到相机背面的距离
    pass


def take_pictures():
    '''拍照'''
    cap = Camera(30)

    count = 0
    while True:
        success, frame = cap.read()
        if not success:
            continue

        cv2.imshow('press c to capture, q to exit', frame)

        key = (cv2.waitKey(1) & 0xFF)

        if key == ord('q'):
            break

        elif key == ord('c'):
            img_path = f'{count}.png'
            cv2.imwrite(img_path, frame)
            print(f'captured at {img_path}')
            count += 1

    cap.release()


if __name__ == '__main__':
    # calibrate_aperture()
    # calibrate_intrinsic_and_distortion()
    take_pictures()
