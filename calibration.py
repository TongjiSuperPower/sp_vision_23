import cv2
import numpy as np
from scipy.spatial.transform import Rotation
from modules.io.mindvision import Camera
from modules.io.robot import Robot
from modules.tools import drawContour, drawPoint, drawAxis, putText, R_gimbal2imu


def calibrate_aperture():
    '''调节光圈使相机能看清5m远装甲板的图案'''
    with Camera(40) as camera:
        while True:
            success, img = camera.read()
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


def calibrate_camera():
    '''
    通过相机标定获得cameraMatrix和distcoeffs
    通过手眼标定获得R_camera2gimbal和t_camera2gimbal
    '''

    patternSize = (10, 7)  # 应该是(row, column)但是会失败，所以(column, row)
    center_distance = 40  # 每个圆心间的距离，单位mm
    centers_3d = np.zeros((patternSize[0]*patternSize[1], 3), np.float32)
    centers_3d[:, :2] = np.mgrid[0:patternSize[0], 0:patternSize[1]].T.reshape(-1, 2)
    centers_3d = centers_3d * center_distance

    img_shape = None
    points_2d = []
    points_3d = []
    R_gripper2base = []
    t_gripper2base = []
    with Robot(30, '/dev/ttyUSB0') as robot:
        count = 0
        while True:
            robot.update()

            img, yaw, pitch = robot.img, robot.yaw, robot.pitch

            if img_shape is None:
                img_shape = img.shape

            img_with_imu = img.copy()
            putText(img_with_imu, f'yaw:{yaw:.2f}, pitch{pitch:.2f}', (100, 100))
            cv2.imshow('Press c to capture, q to finish', img_with_imu)

            key = cv2.waitKey(1) & 0xff
            if key == ord('q'):
                break
            elif key == ord('c'):
                success, centers = cv2.findCirclesGrid(img, patternSize, cv2.CirclesGridFinderParameters_SYMMETRIC_GRID)
                drawing = cv2.drawChessboardCorners(img_with_imu, patternSize, centers, success)

                if success:
                    points_2d.append(centers)
                    points_3d.append(centers_3d)
                    R_gripper2base.append(R_gimbal2imu(yaw, pitch))
                    t_gripper2base.append(np.zeros((3, 1)))

                    count += 1
                    print(f'Successed {count}')

                    putText(drawing, '0', centers[0][0])
                    putText(drawing, '1', centers[1][0])

                cv2.imshow('Press any to continue', drawing)
                cv2.waitKey(0)
                cv2.destroyWindow('Press any to continue')

    if len(points_2d) > 3:
        # 相机标定
        h, w, _ = img_shape
        _, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(points_3d, points_2d, (w, h), None, None)
        print(f'cameraMatrix = np.float32({mtx.tolist()})')
        print(f'distCoeffs = np.float32({dist[0].tolist()})')

        # 手眼标定
        R_target2cam = rvecs
        t_target2cam = tvecs
        R_cam2gripper, t_cam2gripper = cv2.calibrateHandEye(R_gripper2base, t_gripper2base, R_target2cam, t_target2cam)
        print(f'R_camera2gimbal = np.float32({R_cam2gripper.tolist()})')
        print(f't_camera2gimbal = np.float32({t_cam2gripper.tolist()})')

        # 重投影误差
        mean_error = 0
        for i in range(len(points_3d)):
            projected_points, _ = cv2.projectPoints(
                points_3d[i],
                rvecs[i],
                tvecs[i],
                mtx,
                dist
            )
            error = cv2.norm(points_2d[i], projected_points, cv2.NORM_L2) / len(projected_points)
            mean_error += error
        mean_error /= len(points_3d)
        print(f"# 重投影误差: {mean_error:.4f}px")

        # 转换成欧拉角，角度制
        yaw, pitch, roll = Rotation.from_matrix(R_cam2gripper).as_euler('YXZ', degrees=True)
        print(f'# 相机相对于云台: {yaw=:.2f} {pitch=:.2f} {roll=:.2f}')


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
    calibrate: str = None
    while True:
        calibrate = input('光圈/标定/拍照?输入[1/2/3]\n')
        if calibrate == '1' or calibrate == '2' or calibrate == '3':
            break
        else:
            print('请重新输入')

    if calibrate == '1':
        calibrate_aperture()
    elif calibrate == '2':
        calibrate_camera()
    elif calibrate == '3':
        take_pictures()
