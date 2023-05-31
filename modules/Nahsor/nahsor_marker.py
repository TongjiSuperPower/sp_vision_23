# coding:utf-8
import inspect

from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.signal import medfilt

from collections import Counter
from scipy.optimize import curve_fit

import time

from modules.Nahsor.nahsor_utils import *


# 每局比赛旋转方向固定
# 小能量机关的转速固定为16RPM
# 大能量机关转速按照三角函数呈周期性变化。速度目标函数为：speed=a*sin(w*t+b)+c，其中speed的单位为rad/s，t的单位为s
# -----|----------------|-----|-----------------|--> time
#      1   <-- 小符 -->  3     4   <-- 大符 -->   7


class NahsorMarker(object):
    target_centers = []  # 装甲板中心位置
    fit_speeds = []  # 前100次测得的角度或速度
    fit_times = []  # 取得数据的时间

    r_center, radius = None, None  # 拟合圆圆心和半径
    rot_direction = None  # 能量机关转动方向，1为顺时针，-1为逆时针；
    rot_speed = None  # 能量机关转速，单位rad/s

    last_center_for_r = None  # 上一次的装甲板中心
    last_time_for_R = None  # 上次拟合时间
    last_center_for_speed = None
    last_time_for_fit = time.time()
    last_time_for_speed = time.time()  # 上帧图像的时间戳，为了计算两帧图像间的间隔

    def __init__(self, color: COLOR = COLOR.BLUE, fit_speed_mode: FIT_SPEED_MODE = FIT_SPEED_MODE.CURVE_FIT,
                 energy_mode: ENERGY_MODE = ENERGY_MODE.BIG, color_space: COLOR_SPACE = COLOR_SPACE.HSV, target_debug=0,
                 fit_debug=0):
        # def __init__(self, color=COLOR.RED, fit_speed_mode=FIT_SPEED_MODE.BY_SPEED,
        #              energy_mode=ENERGY_MODE.BIG, color_space=COLOR_SPACE.BGR, debug=1):

        self.origin_frame = None  # 原图
        self.energy_mode = energy_mode  # 大符或小符
        self.target_debug = target_debug
        self.fit_debug = fit_debug
        self.fit_speed_mode = fit_speed_mode

        self.show_img = None

        self.speed_func = speed_func

        params = inspect.signature(self.speed_func).parameters
        param_names = list(params.keys())[1:]
        self.speed_param_bounds = [[], []]
        self.speed_params_maxerror = []

        for param_name in param_names:
            if param_name in SPEED_PARAM_BOUNDS:
                self.speed_param_bounds[0].append(SPEED_PARAM_BOUNDS[param_name][0])
                self.speed_param_bounds[1].append(SPEED_PARAM_BOUNDS[param_name][1])
            else:
                self.speed_param_bounds[0].append(-np.inf)
                self.speed_param_bounds[1].append(np.inf)
        self.speed_param_bounds = tuple(self.speed_param_bounds)

        for param_name in param_names:
            if param_name in SPEED_PARAM_MAXERROR:
                self.speed_params_maxerror.append(SPEED_PARAM_MAXERROR[param_name])
            else:
                self.speed_params_maxerror.append(1)
        self.speed_params_maxerror = np.array(self.speed_params_maxerror)
        self.big_start_time = time.time()  # 大符开始的时间(实际只是一个基准，不必此时开始)

        self.speed_params = None  # 速度正弦函数的参数

        if color in COLOR:
            self.color = color  # 目标颜色：红or蓝
        else:
            self.color = COLOR.RED

        if color_space in COLOR_SPACE:
            self.color_space = color_space
        else:
            self.color_space = COLOR_SPACE.HSV

        self.color_range = get_color_range(self.color, self.color_space)

        self.current_center = None  # 目标实际点
        self.target_radius = 0  # 击打半径，已无实际作用
        self.target_corners = None  # pnp解算用点
        self.predict_center = None  # 预测的击打位置

        self.__target_status = STATUS.NOT_FOUND  # 状态指示--> not_found(0), found(1)
        self.__R_status = STATUS.NOT_FOUND  # 圆心状态
        self.fit_status = FIT_STATUS.FAILED  # 是否拟合成功
        self.__fan_change = 0

        self.__pause = 0

    def getTargetStatus(self):
        return self.__target_status

    def binaryzate(self, frame):  # 预处理，进行二值化和初步处理
        img = frame.copy()

        # 二值化开始
        if self.color_space == COLOR_SPACE.HSV:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.color_space == COLOR_SPACE.YUV:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif self.color_space == COLOR_SPACE.SIMU_BLACK:
            cvt_img = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, -150)
            cvt_img = cv2.addWeighted(cvt_img, 2.5, np.zeros(img.shape, img.dtype), 0, 0)
        else:
            cvt_img = img.copy()

        color_range = self.color_range
        lower = color_range[self.color][0]
        upper = color_range[self.color][1]
        mask = cv2.inRange(cvt_img, lower, upper)

        if self.color.name + '_2' in color_range:  # 二次处理,HSV_RED时有
            lower_r2 = color_range[self.color.name + '_2'][0]
            upper_r2 = color_range[self.color.name + '_2'][1]
            mask_r2 = cv2.inRange(cvt_img, lower_r2, upper_r2)
            mask = mask + mask_r2

        # # 二值化结束
        return mask

    def morphological_operation(self, mask):  # 形态学操作
        # # 腐蚀
        # cv2.erode(img, kernel, iterations=1)
        # # 膨胀
        # cv2.dilate(img, kernel, iterations=1)
        # # 先进行腐蚀再进行膨胀就叫做开运算。被用来去除噪音。
        # opening = cv2.morphologyEx(img, cv2.MORPH_OPEN, kernel)
        # # 先膨胀再腐蚀。被用来填充前景物体中的小洞，或者前景上的小黑点。
        # closing = cv2.morphologyEx(img, cv2.MORPH_CLOSE, kernel)
        # # 一幅图像膨胀与腐蚀的差别
        # gradient = cv2.morphologyEx(img, cv2.MORPH_GRADIENT, kernel)
        # # 原始图像与进行开运算之后得到的图像的差。
        # tophat = cv2.morphologyEx(img, cv2.MORPH_TOPHAT, kernel)
        # # 进行闭运算之后得到的图像与原始图像的差
        # cv2.morphologyEx(img, cv2.MORPH_BLACKHAT, kernel)
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))  # 方形
        # cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2))  # 椭圆
        # cv2.getStructuringElement(cv2.MORPH_CROSS, (2, 2))  # 十字形

        # mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2)))
        # mask = cv2.medianBlur(mask, 5)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)), 2)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)), 2)
        # mask = cv2.medianBlur(mask, 3)

        return mask

    def mark(self, frame):
        if self.__pause == 1:
            return
        # mark之前先初始化
        self.origin_frame = frame
        # self.target_radius = 0  # 击打半径
        # self.predict_center = None  # 预测的击打位置

        self.__target_status = STATUS.NOT_FOUND  # 状态指示--> found(1) and not_found(0)

        # ----------- 图像预处理 start -----------
        mask = cv2.blur(frame, (3, 3))
        mask = self.binaryzate(mask)
        mask = self.morphological_operation(mask)
        self.show_img = mask
        # if self.target_debug:
        #     cv2.namedWindow('mask', 0)
        #     cv2.resizeWindow('mask', int(1200 * (800 - 80) / 800), 800 - 80)
            # cv2.imshow('mask', mask)
        # ----------- 图像预处理 end ------------
        # ----------- 寻找目标 start ----------
        # 获取轮廓
        mask_cp = mask.copy()
        contours, hierarchy = cv2.findContours(mask_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        parent_contours = get_parent_contours(contours, hierarchy)
        target_contour = get_target_fan(contours, parent_contours)

        if target_contour is not None:

            # 找到棒棒糖上半部分的中心点
            M = cv2.moments(target_contour)
            cx, cy = int(M['m10'] / M['m00']), int(M['m01'] / M['m00'])
            fan_rect = cv2.minAreaRect(target_contour)
            # 将中心点坐标四舍五入为整数
            target_center = (cx, cy)

            self.last_center_for_r = self.current_center
            self.current_center = target_center
            if self.r_center is not None:
                self.target_corners = get_pnp_points(target_contour, self.r_center)
            self.target_radius = (fan_rect[1][0] + fan_rect[1][1]) / (2 * 4)  # 已经无实际作用
            self.__target_status = STATUS.FOUND
        else:
            self.__target_status = STATUS.NOT_FOUND
        ##############
        if self.target_debug:
            orig1 = self.origin_frame.copy()
            if contours is not None:
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) < CONTOUR_MIN_AREA:
                        continue
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    orig1 = cv2.drawContours(orig1, [box], 0, (0, 255, 0), 3)
                    orig1 = cv2.putText(orig1, str(cv2.contourArea(contour)), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX,
                                        1, (0, 255, 0), 2)

            # 绘制多边形
            if self.r_center is not None and self.target_corners is not None:
                for i, p in enumerate(self.target_corners):
                    point = p
                    dx = point[0] - self.r_center[0]
                    dy = point[1] - self.r_center[1]
                    angle = np.arctan2(dy, dx) * 180 / np.pi
                    # orig1 = cv2.putText(orig1, f'{angle:.0f}', (p[0] - 20, p[1] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1,
                    #                     (0, 255, 255), 5)
                    orig1 = cv2.putText(orig1, f'{i}', tuple(point), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 5)

                    orig1 = cv2.circle(orig1, tuple(point), 1, (0, 255, 255), 10)
            # cv2.namedWindow('contours', 0)
            # cv2.resizeWindow('contours', int(1200 * (800 - 80) / 800), 800 - 80)
            # cv2.imshow('contours', orig1)
            self.show_img = orig1

        ##############
        # ----------- 寻找目标 end ----------

        if self.__target_status == STATUS.FOUND:
            # ----------- 寻找圆心的位置 start -----------
            if get_distance(self.current_center, self.last_center_for_r) > CENTER_MAX_DISTANCE:
                self.__R_status = STATUS.NOT_FOUND
                self.__fan_change = 1
                print('fan error')

            self.set_target_centers()
            # if self.__R_status == STATUS.NOT_FOUND:
            R_by_contours = get_r_by_contours(contours, hierarchy, self.current_center,
                                              self.target_radius)
            # R_by_centers = get_r_by_centers(self.target_centers)

            # if 0 <= get_distance(R_by_contours, R_by_centers) < R_MAX_DISTANCE:
            if R_by_contours is not None:
                self.r_center = R_by_contours
            # self.__R_status = STATUS.FOUND
            # else:
            #     if R_by_contours is not None:
            #         self.r_center = R_by_contours
            #     elif R_by_centers is not None:
            #         self.r_center = R_by_centers
            #     self.__R_status = STATUS.NOT_FOUND
            # self.last_time_for_R = time.time()
            # else:
            #     if time.time() - self.last_time_for_R > FIND_R_INTERVAL:
            #         R_by_centers = get_r_by_centers(self.target_centers)
            #         if get_distance(self.r_center, R_by_centers) > R_MAX_DISTANCE or \
            #                 get_distance(self.current_center, self.last_center_for_r) > CENTER_MAX_DISTANCE:
            #             self.__R_status = STATUS.NOT_FOUND
            # ----------- 寻找圆心的位置 end -----------

            # ----------- 预测 start -----------------
            if USE_PREDICT == True:
                if self.energy_mode == ENERGY_MODE.BIG:
                    # if self.__r_change == 0 and self.last_center is not None:
                    if self.__target_status == STATUS.NOT_FOUND or self.r_center is None and self.fit_status != FIT_STATUS.SUCCESS:
                        self.fit_status = FIT_STATUS.FAILED
                    elif self.__fan_change == 1:
                        # 需要调整上一个点的位置

                        self.last_time_for_speed = time.time()
                        self.last_center_for_speed = self.current_center
                        self.__fan_change = 0
                    elif self.fit_status == FIT_STATUS.FAILED:
                        self.fit_status = FIT_STATUS.FITTING

                        self.last_time_for_speed = time.time()
                        self.last_center_for_speed = self.current_center
                    elif self.fit_status == FIT_STATUS.FITTING:
                        if time.time() - self.last_time_for_speed > FIT_INTERVAL:
                            self.set_fit_speeds()

                            if len(self.fit_speeds) > FIT_MIN_LEN:
                                speed_params, speed_cov = self.fit_speed_params()
                                if speed_cov is None:
                                    print("speed_cov is None")
                                else:
                                    diag_of_speed_cov = np.diag(speed_cov)
                                    speed_err = np.sqrt(diag_of_speed_cov)
                                    print("error: ", speed_err)
                                    if (np.all(speed_err < self.speed_params_maxerror) and len(
                                            self.fit_speeds) > 5 * FIT_MIN_LEN):
                                        self.fit_status = FIT_STATUS.SUCCESS
                                        self.last_time_for_fit = time.time()
                                    self.speed_params = speed_params

                    elif self.fit_status == FIT_STATUS.SUCCESS:
                        pass
                        # if time.time() - self.last_time_for_fit > SPEED_REFIT_INTERVAL:
                        #     self.fit_status = FIT_STATUS.FITTING
                        # if time.time() - self.last_time_for_speed > FIT_INTERVAL:
                        #     self.set_fit_speeds()

                    # if self.fit_params is not None:
                    #     self.rot_speed = speed_func(time.time() - self.big_start_time, self.fit_params[0],
                    #                                 self.fit_params[1], self.fit_params[2],
                    #                                 self.fit_params[3])
                else:
                    self.rot_speed = SMALL_ROT_SPEED * 2 * np.pi / 60  # RPM->rad/s

                # if self.__R_status == STATUS.FOUND and len(self.target_centers) > FIT_MIN_LEN:
                if self.fit_status != FIT_STATUS.SUCCESS and self.rot_speed is not None and len(
                        self.target_centers) > FIT_MIN_LEN and self.r_center is not None:
                    clockwise1 = get_clockwise(self.r_center, self.target_centers[-4],
                                               self.target_centers[-1])
                    if self.rot_direction is None or self.rot_direction != clockwise1:
                        self.rot_direction = self.get_rot_direction()

    def markFrame(self):
        """
        画图函数
        :return:一帧标记好的图像
        """
        orig = self.origin_frame.copy()
        if self.current_center is not None:
            p = self.current_center
            orig = cv2.circle(orig, (int(p[0]), int(p[1])), int(self.target_radius), (0, 255, 255), 3)
        if self.rot_speed is not None:
            orig = cv2.putText(orig, str(round(self.rot_speed, 3)),
                               (int(self.current_center[0]), int(self.current_center[1])), cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 255), 2)
        if self.predict_center is not None:
            p = self.predict_center
            orig = cv2.circle(orig, (int(p[0]), int(p[1])), int(self.target_radius), (240, 32, 160), 3)
            # orig = cv2.putText(orig, 'predict', (int(self.current_center[0]), int(self.current_center[1])),
            #                     cv2.FONT_HERSHEY_SIMPLEX, 1, (240, 32, 160), 2)

        if self.r_center is not None:
            orig = cv2.circle(orig, (int(self.r_center[0]), int(self.r_center[1])), 5, (0, 255, 0), 3)
        if self.target_centers is not None:
            for p in self.target_centers:
                orig = cv2.circle(orig, (int(p[0]), int(p[1])), 3, (0, 255, 0), 1)

        if self.__target_status == STATUS.FOUND:
            s = 'found'
        else:
            s = 'not found'
        orig = cv2.putText(orig, self.color.name + ' ' + s, (0, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.3,
                           (0, 255, 0), 4)

        return orig

    def get_theta(self, predict_time):
        """
        返回下一个点与当前点的角度
        """
        if self.energy_mode == ENERGY_MODE.BIG:
            if self.speed_params is not None:
                fit_param_args = tuple(self.speed_params[0:4])

                # angle = self.get_predict_time() * speed_func(
                #     time.time() - self.big_start_time + self.get_predict_time(),
                #     *fit_param_args)
                # angle = self.get_predict_time() * self.rot_speed
                # angle, _ = quad(speed_func, time.time() - self.big_start_time,
                #                 time.time() + self.get_predict_time() - self.big_start_time, args=fit_param_args)
                angle = angle_func(time.time() + predict_time - self.big_start_time,
                                   *fit_param_args) - angle_func(time.time() - self.big_start_time,
                                                                 *fit_param_args)
            else:
                angle = predict_time * self.rot_speed
                # angle = 0
        else:
            angle = predict_time * self.rot_speed
        return angle * 180 / np.pi

    def set_2d_predict_center(self, predict_time):
        if USE_PREDICT == 0:
            self.predict_center = self.current_center
        else:
            if self.rot_direction is not None:
                theta = self.rot_direction * self.get_theta(predict_time)
                rot_mat = cv2.getRotationMatrix2D(self.r_center, theta, 1)
                sinA = rot_mat[0][1]
                cosA = rot_mat[0][0]

                xx = -(self.r_center[0] - self.current_center[0])
                yy = -(self.r_center[1] - self.current_center[1])

                self.predict_center = (
                    int(self.r_center[0] + cosA * xx - sinA * yy), int(self.r_center[1] + sinA * xx + cosA * yy))

    def get_2d_predict_corners(self, predict_time):
        if USE_PREDICT == 0:
            return self.target_corners
        else:
            if self.rot_direction is not None:
                theta = self.rot_direction * self.get_theta(predict_time)
                rot_mat = cv2.getRotationMatrix2D(self.r_center, theta, 1)

                def rotate(rot_mat, r_center, point):
                    sinA = rot_mat[0][1]
                    cosA = rot_mat[0][0]
                    xx = -(r_center[0] - point[0])
                    yy = -(r_center[1] - point[1])
                    return (int(r_center[0] + cosA * xx - sinA * yy), int(r_center[1] + sinA * xx + cosA * yy))

                return np.array([rotate(rot_mat, self.r_center, point) for point in self.target_corners])
            else:
                return self.target_corners

    def get_rot_direction(self):
        rot_directions = []
        for i in range(len(self.target_centers) - 2):
            rot_directions.append(get_clockwise(self.r_center, self.target_centers[i],
                                                self.target_centers[i + 2]))
        counter = Counter(rot_directions)
        return counter.most_common(1)[0][0]

    # 拆出计算速度的部分，如果误差过大重新预测
    def set_fit_speeds(self):
        current_time = time.time()
        interval = current_time - self.last_time_for_speed
        rot_angle = angle_between_points(self.r_center, self.current_center,
                                         self.last_center_for_speed)
        self.rot_speed = rot_angle / interval

        self.fit_speeds.append(self.rot_speed)
        self.fit_times.append(current_time - self.big_start_time)
        if len(self.fit_speeds) > FIT_MAX_LEN:
            self.fit_speeds.pop(0)
            self.fit_times.pop(0)

        self.last_time_for_speed = current_time
        self.last_center_for_speed = self.current_center

    def set_target_centers(self):
        self.target_centers.append(self.current_center)
        if len(self.target_centers) > TARGET_CENTERS_LEN:
            self.target_centers.pop(0)

    def fit_speed_params(self):
        # fit_times = add_list(np.array(times))
        fit_times = np.array(self.fit_times)

        # smooth_data = gaussian_filter1d(self.fit_speeds, sigma=1)
        # smooth_data = medfilt(self.fit_speeds)
        # smooth_data = medfilt(smooth_data)
        smooth_data = self.fit_speeds
        # smooth_data = gaussian_filter1d(self.fit_speeds, sigma=1)
        # smooth_data = medfilt(smooth_data)
        # smooth_data = medfilt(smooth_data)
        smooth_data = gaussian_filter1d(smooth_data, sigma=1)

        bounds = self.speed_param_bounds
        if self.speed_params is None:
            p0 = []
            for i in range(len(bounds[0])):
                if not np.isinf(bounds[0][i]) and not np.isinf(bounds[1][i]):
                    p0.append((bounds[0][i] + bounds[1][i]) / 2)
                else:
                    p0.append(1)
        else:
            p0 = self.speed_params

        try:
            speed_params, speed_cov = curve_fit(self.speed_func, fit_times, smooth_data, maxfev=1000,
                                                bounds=bounds, p0=p0)
        except Exception as e:
            speed_params, speed_cov = None, None

        # 最小二乘法拟合
        # def residual(params, t, target_speed):
        #     a, w, b, c = params
        #     return speed_func(t, a, w, b, c) - target_speed
        #
        # speed_params, cov = leastsq(residual, [1, 1, 1, 1], args=(fit_times, smooth_angles))

        if self.fit_debug == 1 and speed_params is not None:
            plt.figure(figsize=(30, 10), dpi=100, num=1)
            plt.clf()
            plt.plot(fit_times, self.fit_speeds, alpha=0.8, linewidth=1)
            plt.plot(fit_times, smooth_data, alpha=0.8, linewidth=1)
            predict_args = tuple(speed_params)
            predict_data = [self.speed_func(add_time, *predict_args)
                            for add_time in fit_times]
            plt.plot(fit_times, predict_data, alpha=0.8, linewidth=1)
            plt.pause(0.00001)
            print("Speed paras:", speed_params)
            print('speed cov:', speed_cov)
        return speed_params, speed_cov

    def show_fit_result(self):
        if self.fit_debug == 1:
            add_times = np.array(self.fit_times)
            smooth_data = gaussian_filter1d(self.fit_speeds, sigma=1)
            smooth_data = gaussian_filter1d(smooth_data, sigma=1)

            plt.figure(figsize=(30, 10), dpi=100, num=1)
            plt.clf()
            plt.plot(add_times, self.fit_speeds, alpha=0.8, linewidth=1)
            plt.plot(add_times, smooth_data, alpha=0.8, linewidth=10, marker='x')
            predict_args = tuple(self.speed_params)
            predict_data = [self.speed_func(add_time, *predict_args)
                            for add_time in add_times]
            plt.plot(add_times, predict_data, alpha=0.8, linewidth=1)
            # plt.plot(np.arange(1, 13, 0.1), func(np.arange(1, 13, 0.1)), alpha=0.8, linewidth=2, marker='o')
            plt.pause(0.0001)
            # print("Speed paras:", speed_params)
            # print('speed cov:', speed_cov)

    def resume(self):
        if self.__pause == 0:
            self.__pause = 1
            self.target_centers = []
            self.r_center = None
            self.__target_status = STATUS.NOT_FOUND
            self.current_center = None
            self.__fan_change = 0
            self.fit_status = FIT_STATUS.FAILED
            self.last_center_for_r = None
            self.last_time_for_R = None
            self.predict_center = None
            self.__R_status = STATUS.NOT_FOUND
        else:
            self.__pause = 0
