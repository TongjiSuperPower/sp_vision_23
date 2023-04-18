# coding:utf-8
import inspect

from scipy.ndimage import gaussian_filter1d

from collections import Counter
from scipy.optimize import curve_fit

import time
from Utils import *

# 每局比赛旋转方向固定
# 小能量机关的转速固定为10RPM
# 大能量机关转速按照三角函数呈周期性变化。速度目标函数为：speed=a*sin(w*t+b)+c，其中speed的单位为rad/s，t的单位为s
# -----|----------------|-----|-----------------|--> time
#      1   <-- 小符 -->  3     4   <-- 大符 -->   7


class NahsorMarker(object):
    target_centers = []  # 装甲板中心位置
    fit_data = []  # 前100次测得的角度或速度
    fit_times = []  # 取得数据的时间

    r_center, radius = None, 0  # 拟合圆圆心和半径
    rot_direction = 1  # 能量机关转动方向，1为顺时针，-1为逆时针；
    rot_speed = 0  # 能量机关转速，单位rad/s

    last_center_for_r = None  # 上一次的装甲板中心
    last_time_for_fit_r = None  # 上次拟合时间
    last_center_for_fitdata = None
    last_time_for_predict = time.time()
    last_time_for_fitdata = time.time()  # 上帧图像的时间戳，为了计算两帧图像间的间隔

    def __init__(self, color: COLOR = COLOR.RED, fit_speed_mode: FIT_SPEED_MODE = FIT_SPEED_MODE.BY_SPEED,
                 energy_mode: ENERGY_MODE = ENERGY_MODE.BIG, color_space: COLOR_SPACE = COLOR_SPACE.BGR, find_debug=0,
                 fit_debug=0):
        # def __init__(self, color=COLOR.RED, fit_speed_mode=FIT_SPEED_MODE.BY_SPEED,
        #              energy_mode=ENERGY_MODE.BIG, color_space=COLOR_SPACE.BGR, debug=1):

        self.origin_frame = None  # 原图
        self.energy_mode = energy_mode  # 大符或小符
        self.find_debug = find_debug
        self.fit_debug = fit_debug
        self.fit_speed_mode = fit_speed_mode
        if self.fit_speed_mode == FIT_SPEED_MODE.BY_SPEED:
            self.fit_func = speed_func
        else:
            self.fit_func = angle_func
        params = inspect.signature(self.fit_func).parameters
        param_names = list(params.keys())[1:]
        self.speed_param_bounds = [[], []]
        self.speed_param_maxerror = []
        # self.speed_params_min = []
        # self.speed_params_max = []
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
                self.speed_param_maxerror.append(SPEED_PARAM_MAXERROR[param_name])
            else:
                self.speed_param_maxerror.append(1)
        self.speed_param_maxerror = np.array(self.speed_param_maxerror)
        self.big_start_time = time.time()  # 大符开始的时间(实际只是一个基准，不必此时开始)

        self.fit_params = None  # 速度正弦函数的参数

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
        self.target_radius = 0  # 击打半径
        self.predict_center = None  # 预测的击打位置

        self.__target_status = TARGET_STATUS.NOT_FOUND  # 状态指示--> not_found(0), found(1), fitting(2)
        self.__r_change = 0  # 圆心是否变化，0：未改变 / 1：改变
        self.__fit_speed_status = FIT_SPEED_STATUS.FAILED  # 是否拟合成功

    def __preprocess(self, frame):  # 预处理，进行二值化和初步处理
        img = frame.copy()

        # 二值化开始
        if self.color_space == COLOR_SPACE.HSV:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        elif self.color_space == COLOR_SPACE.YUV:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
        elif self.color_space == COLOR_SPACE.SIMU_BLACK:
            cvt_img = cv2.addWeighted(img, 1, np.zeros(img.shape, img.dtype), 0, -150)
            cvt_img = cv2.addWeighted(cvt_img, 2.5, np.zeros(img.shape, img.dtype), 0, 0)
            # cvt_img = cv2.convertScaleAbs(cvt_img, alpha=1)

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
        # kernel1 = cv2.getStructuringElement(cv2.MORPH_RECT, (2, 2))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        # mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        # # 获取矩形形状的结构元素
        # 对掩膜进行开运算，先进行腐蚀再进行膨胀，去除噪声和小的斑点
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (CORE_SIZE, CORE_SIZE)))
        # mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (2, 2)))
        # mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel1)
        # # 对掩膜进行闭运算，先进行膨胀再进行腐蚀，连接目标区域的间断
        # # 对掩膜进行膨胀操作，填充目标区域的空洞
        # mask = cv2.dilate(mask, kernel)

        mask = cv2.dilate(mask, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.erode(mask, cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (4, 4)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3)))
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, cv2.getStructuringElement(cv2.MORPH_RECT, (4, 4)))
        # return cvt_img
        return mask

    # def __find_target1(self, mask):
    #     # 获取轮廓
    #     mask_cp = mask.copy()
    #     contours, hierarchy = cv2.findContours(mask_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)
    #     # ----------- 按约束条件筛选轮廓 start -----------
    #
    #     parent_contours = {}
    #     # 寻找有方形子轮廓的轮廓
    #     for i, contour in enumerate(contours):
    #         rect = cv2.minAreaRect(contour)
    #         w_armor = max(rect[1][0], rect[1][1])
    #         h_armor = min(rect[1][0], rect[1][1])
    #         if cv2.contourArea(contour) < MIN_AREA:
    #             continue
    #         if hierarchy[0][i][3] != -1 and SQUARE_WH_RATIO[0] < w_armor / h_armor < SQUARE_WH_RATIO[1]:
    #             try:
    #                 parent_contours[hierarchy[0][i][3]].append(i)
    #             except KeyError:
    #                 pass
    #         else:
    #             parent_contours[i] = []
    #
    #     # 统计子轮廓数量，并记录子轮廓数量为1且面积最大的轮廓
    #     max_area = float('-inf')
    #     target_contour = None
    #     for father_contour_number, child_contours in parent_contours.items():
    #         num_sub_contours = len(child_contours)
    #         if num_sub_contours == 1 and cv2.contourArea(contours[father_contour_number]) > max_area:
    #             max_area = cv2.contourArea(contours[father_contour_number])
    #             target_contour = contours[parent_contours[father_contour_number][0]]
    #
    #     return target_contour, contours

    # def __find_hit_rect(self, armor_contour):
    #     if self.r_center is None:
    #         return (None, None), (None, None), None
    #     # 计算中心点
    #     armor_center, armor_size, armor_angle = cv2.minAreaRect(armor_contour)
    #
    #     # 计算平移距离，由装甲板中心点与圆心距离确定
    #     MOVE_RATIO = 0.2  # 比例由图纸尺寸决定
    #     delta_x = armor_center[0] - self.r_center[0]
    #     delta_y = armor_center[1] - self.r_center[1]
    #     radius = math.sqrt(delta_x * delta_x + delta_y * delta_y)
    #     dx, dy = MOVE_RATIO * delta_x, MOVE_RATIO * delta_y
    #     # 计算出击打点的中心点坐标
    #     hit_center_x, hit_center_y = armor_center[0] + dx, armor_center[1] + dy
    #
    #     # 计算直线斜率
    #     armor_slope = (armor_center[1] - self.r_center[1]) / (armor_center[0] - self.r_center[0])
    #
    #     # 计算角度
    #     hit_rect_angle = math.atan(armor_slope) * 180 / math.pi
    #
    #     # 将角度转换为顺时针方向的角度
    #     if hit_rect_angle > 0:
    #         hit_rect_angle -= 90
    #     else:
    #         hit_rect_angle += 270
    #
    #     # 计算装甲板长宽，由上半灯条确定
    #     W_RATIO, H_RATIO = 0.5, 1
    #     hit_rect_size = (int(W_RATIO * radius), int(H_RATIO * radius))
    #     # RotateRect格式
    #     return (hit_center_x, hit_center_y), hit_rect_size, hit_rect_angle

    def mark(self, frame):
        # mark之前先初始化
        self.origin_frame = frame
        # self.target_radius = 0  # 击打半径
        # self.predict_center = None  # 预测的击打位置

        self.__target_status = TARGET_STATUS.NOT_FOUND  # 状态指示--> found(1) and not_found(0)

        # ----------- 图像预处理 start -----------
        mask = self.__preprocess(frame)
        if self.find_debug:
            cv2.namedWindow('mask', 0)
            cv2.resizeWindow('mask', int(1200 * (800 - 80) / 800), 800 - 80)
            cv2.imshow('mask', mask)
        # ----------- 图像预处理 end ------------
        # ----------- 寻找目标 start ----------
        # 获取轮廓
        mask_cp = mask.copy()
        contours, hierarchy = cv2.findContours(mask_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        parent_contours = get_parent_contours(contours, hierarchy)
        target_contour = get_target(contours, parent_contours)

        if target_contour is not None:
            target_center, target_size, target_angle = cv2.minAreaRect(target_contour)
            self.last_center_for_r = self.current_center
            self.current_center = target_center

            self.target_radius = (target_size[0] + target_size[1]) / 2
            self.__target_status = TARGET_STATUS.FOUND
        else:
            self.__target_status = TARGET_STATUS.NOT_FOUND
        ##############
        if self.find_debug:
            orig1 = self.origin_frame.copy()
            if contours is not None:
                for i, contour in enumerate(contours):
                    if cv2.contourArea(contour) < CONTOUR_MIN_AREA:
                        continue
                    rect = cv2.minAreaRect(contour)
                    box = cv2.boxPoints(rect)
                    box = np.int0(box)
                    orig1 = cv2.drawContours(orig1, [box], 0, (0, 255, 0), 3)
                    orig1 = cv2.putText(orig1, str(i), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.namedWindow('contours', 0)
            cv2.resizeWindow('contours', int(1200 * (800 - 80) / 800), 800 - 80)
            cv2.imshow('contours', orig1)
        ##############
        # ----------- 寻找目标 end ----------

        if self.__target_status == TARGET_STATUS.FOUND:
            # ----------- 寻找圆心的位置 start -----------
            if get_distance(self.current_center, self.last_center_for_r) > CENTER_MAX_DISTANCE:
                self.__r_change = 1

            if self.__r_change:
                r_center_by_contours = get_r_by_contours(contours, parent_contours, self.current_center,
                                                         self.target_radius)
                self.set_target_centers()
                r_center_by_centers = get_r_by_centers(self.target_centers)

                if get_distance(r_center_by_contours, r_center_by_centers) < R_MAX_DISTANCE:
                    self.r_center = r_center_by_contours
                    self.__r_change = 0
                else:
                    if r_center_by_contours is not None:
                        self.r_center = r_center_by_contours
                    elif r_center_by_centers is not None:
                        self.r_center = r_center_by_centers
                    self.__r_change = 1
                self.last_time_for_fit_r = time.time()
            else:
                self.set_target_centers()
                if time.time() - self.last_time_for_fit_r > R_REFIT_INTERVAL:
                    r_center_by_centers = get_r_by_centers(self.target_centers)
                    if get_distance(self.r_center, r_center_by_centers) > R_MAX_DISTANCE or \
                            get_distance(self.current_center, self.last_center_for_r) > CENTER_MAX_DISTANCE:
                        self.__r_change = 1
            # ----------- 寻找圆心的位置 end -----------

            # ----------- 预测 start -----------------
            if USE_PREDICT == 0:
                self.predict_center = self.current_center
            else:
                if self.energy_mode == ENERGY_MODE.BIG:
                    # if self.__r_change == 0 and self.last_center is not None:
                    if self.__target_status == TARGET_STATUS.NOT_FOUND or self.r_center is None:
                        self.__fit_speed_status = FIT_SPEED_STATUS.FAILED
                    elif self.__fit_speed_status == FIT_SPEED_STATUS.FAILED:
                        self.__fit_speed_status = FIT_SPEED_STATUS.FITTING
                        # self.big_start_time = time.time()
                        # self.fit_data = []
                        # self.fit_times = []
                        # self.last_time_for_fitdata = self.big_start_time
                        self.last_time_for_fitdata = time.time()
                        self.last_center_for_fitdata = self.current_center
                    elif self.__fit_speed_status == FIT_SPEED_STATUS.FITTING:
                        if time.time() - self.last_time_for_fitdata > FIT_INTERVAL:
                            self.set_fit_data()

                            if len(self.fit_data) > FIT_MIN_LEN:
                                speed_params, speed_cov = self.fit_speed_params()
                                speed_err = np.sqrt(np.diag(speed_cov))
                                print("error: ", speed_err)
                                if np.all(speed_err < self.speed_param_maxerror) and len(self.fit_data) > 5 * FIT_MIN_LEN:
                                    self.__fit_speed_status = FIT_SPEED_STATUS.SUCCESS
                                    self.last_time_for_predict = time.time()
                                self.fit_params = speed_params

                    elif self.__fit_speed_status == FIT_SPEED_STATUS.SUCCESS:
                        if time.time() - self.last_time_for_predict > SPEED_REFIT_INTERVAL:
                            self.__fit_speed_status = FIT_SPEED_STATUS.FITTING
                        if time.time() - self.last_time_for_fitdata > FIT_INTERVAL:
                            self.set_fit_data()

                    # if self.fit_params is not None:
                    #     self.rot_speed = speed_func(time.time() - self.big_start_time, self.fit_params[0],
                    #                                 self.fit_params[1], self.fit_params[2],
                    #                                 self.fit_params[3])
                else:
                    self.rot_speed = SMALL_ROT_SPEED * 2 * np.pi / 60  # RPM->rad/s

                if self.__r_change == 0 and len(self.target_centers) > FIT_MIN_LEN:
                    clockwise1 = get_clockwise(self.r_center, self.target_centers[-4],
                                               self.target_centers[-1])
                    if self.rot_direction is None or self.rot_direction != clockwise1:
                        self.rot_direction = self.get_rot_direction()

                if self.rot_direction is not None:
                    self.predict_center = self.get_predict_center()

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

        if self.__target_status == TARGET_STATUS.FOUND:
            s = 'found'
        else:
            s = 'not found'
        orig = cv2.putText(orig, self.color.name + ' ' + s, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)

        return orig

    def get_predict_time(self):
        """
        返回向后预测的时间
        """
        return DELAY_TIME

    def get_theta(self):
        """
        返回下一个点与当前点的角度
        """
        if self.energy_mode == ENERGY_MODE.BIG:
            if self.fit_params is not None:
                fit_param_args = tuple(self.fit_params[0:4])

                # angle = self.get_predict_time() * speed_func(
                #     time.time() - self.big_start_time + self.get_predict_time(),
                #     *fit_param_args)
                # angle = self.get_predict_time() * self.rot_speed
                # angle, _ = quad(speed_func, time.time() - self.big_start_time,
                #                 time.time() + self.get_predict_time() - self.big_start_time, args=fit_param_args)
                angle = angle_func(time.time() + self.get_predict_time() - self.big_start_time,
                                   *fit_param_args) - angle_func(time.time() - self.big_start_time,
                                                                 *fit_param_args)
            else:
                angle = self.get_predict_time() * self.rot_speed
                # angle = 0
        else:
            angle = self.get_predict_time() * self.rot_speed
        return angle * 180 / np.pi

    def get_predict_center(self):
        theta = self.rot_direction * self.get_theta()
        rot_mat = cv2.getRotationMatrix2D(self.r_center, theta, 1)
        sinA = rot_mat[0][1]
        cosA = rot_mat[0][0]

        xx = -(self.r_center[0] - self.current_center[0])
        yy = -(self.r_center[1] - self.current_center[1])

        return (
            int(self.r_center[0] + cosA * xx - sinA * yy),
            int(self.r_center[1] + sinA * xx + cosA * yy))

    def get_rot_direction(self):
        rot_directions = []
        for i in range(len(self.target_centers) - 2):
            rot_directions.append(get_clockwise(self.r_center, self.target_centers[i],
                                                self.target_centers[i + 2]))
        counter = Counter(rot_directions)
        return counter.most_common(1)[0][0]


# 拆出计算速度的部分，如果误差过大重新预测
    def set_fit_data(self):
        current_time = time.time()
        interval = current_time - self.last_time_for_fitdata
        rot_angle = angle_between_points(self.r_center, self.current_center,
                                         self.last_center_for_fitdata)
        self.rot_speed = rot_angle / interval
        if self.fit_speed_mode == FIT_SPEED_MODE.BY_SPEED:
            data = self.rot_speed
        else:
            if len(self.fit_data) > 0:
                data = self.fit_data[-1] + rot_angle
            else:
                data = rot_angle
        self.fit_data.append(data)
        self.fit_times.append(current_time - self.big_start_time)
        if len(self.fit_data) > FIT_MAX_LEN:
            self.fit_data.pop(0)
            self.fit_times.pop(0)

        self.last_time_for_fitdata = current_time
        self.last_center_for_fitdata = self.current_center

    def set_target_centers(self):
        self.target_centers.append(self.current_center)
        if len(self.target_centers) > TARGET_CENTERS_LEN:
            self.target_centers.pop(0)

    def fit_speed_params(self):
        # add_times = add_list(np.array(times))
        add_times = np.array(self.fit_times)
        smooth_data = gaussian_filter1d(self.fit_data, sigma=1)
        smooth_data = gaussian_filter1d(smooth_data, sigma=1)
        # smooth_angles = 10 * smooth_angles
        # angles = add_list(np.array(angles))
        # angles = smooth_data(angles)
        # angles = wavelet_noising(angles)

        # bounds = (self.speed_params_min, self.speed_params_max)
        bounds = self.speed_param_bounds
        if self.fit_params is None:
            p0 = []
            for i in range(len(bounds[0])):
                if not np.isinf(bounds[0][i]) and not np.isinf(bounds[1][i]):
                    p0.append((bounds[0][i] + bounds[1][i]) / 2)
                else:
                    p0.append(1)
        else:
            p0 = self.fit_params

        speed_params, speed_cov = curve_fit(self.fit_func, add_times, smooth_data, maxfev=10000,
                                                     bounds=bounds, p0=p0)

        # 最小二乘法拟合
        # def residual(params, t, target_speed):
        #     a, w, b, c = params
        #     return speed_func(t, a, w, b, c) - target_speed
        #
        # speed_params, cov = leastsq(residual, [1, 1, 1, 1], args=(add_times, smooth_angles))

        if self.fit_debug == 1:
            # self.visualizer.plot(())
            plt.figure(figsize=(30, 10), dpi=100, num=1)
            plt.clf()
            plt.plot(add_times, self.fit_data, alpha=0.8, linewidth=1)
            plt.plot(add_times, smooth_data, alpha=0.8, linewidth=10, marker='x')
            predict_args = tuple(speed_params)
            predict_data = [self.fit_func(add_time, *predict_args)
                            for add_time in add_times]
            plt.plot(add_times, predict_data, alpha=0.8, linewidth=1)
            # plt.plot(np.arange(1, 13, 0.1), func(np.arange(1, 13, 0.1)), alpha=0.8, linewidth=2, marker='o')
            plt.pause(0.00001)
            print("Speed paras:", speed_params)
            print('speed cov:', speed_cov)
        return speed_params, speed_cov

    def show_fit_result(self):
        if self.fit_debug == 1:
            add_times = np.array(self.fit_times)
            smooth_data = gaussian_filter1d(self.fit_data, sigma=1)
            smooth_data = gaussian_filter1d(smooth_data, sigma=1)

            plt.figure(figsize=(30, 10), dpi=100, num=1)
            plt.clf()
            plt.plot(add_times, self.fit_data, alpha=0.8, linewidth=1)
            plt.plot(add_times, smooth_data, alpha=0.8, linewidth=10, marker='x')
            predict_args = tuple(self.fit_params)
            predict_data = [self.fit_func(add_time, *predict_args)
                            for add_time in add_times]
            plt.plot(add_times, predict_data, alpha=0.8, linewidth=1)
            # plt.plot(np.arange(1, 13, 0.1), func(np.arange(1, 13, 0.1)), alpha=0.8, linewidth=2, marker='o')
            plt.pause(0.0001)
            # print("Speed paras:", speed_params)
            # print('speed cov:', speed_cov)
