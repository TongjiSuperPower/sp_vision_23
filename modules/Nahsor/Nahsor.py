import cv2
import time
from NahsorConfig import *
from Utils import *


# 每局比赛旋转方向固定
# 小能量机关的转速固定为10RPM
# 大能量机关转速按照三角函数呈周期性变化。速度目标函数为：spd=0.785*sin(1.884*t)+1.305，其中spd的单位为rad/s，t的单位为s

# -----|----------------|-----|-----------------|--> time
#      1   <-- 小符 -->  3     4   <-- 大符 -->   7


class NahsorMarker(object):
    last_detec_time = time.time()  # 上次进行转向检测的时间
    big_start_time = None  # 大符开始的时间
    popt = None  # 速度正弦函数的参数
    jump_flag = True

    fit_circle_points = []
    r_center, radius = [None] * 2  # 拟合圆圆心和半径

    last_point = None  # 上一次的装甲板中心
    last_calc_point = None
    last_r_center = None  # 上一次的R标坐标
    last_time = time.time()  # 上帧图像的时间戳，为了计算两帧图像间的间隔
    last_spd = 13.0  # 上一次测得的速度
    last_interval = 0
    last_points = []
    speeds = []  # 前100次测得的速度
    intervals = []  # 前100次测速的时间间隔
    fit_curve_success = False  # 拟合速度曲线成功标志位

    rot_direction = []  # 能量机关转动方向，0为顺时针，1为逆时针；
    rot_spd = 10.0  # 能量机关转速，单位转/min (RPM)

    def __init__(self, color: str, init_v=20, debug=0, get_R_method=0):
        self.__debug = debug
        self.__get_r_method = get_R_method
        self.__init_v = init_v
        self.__frame = None
        self.__color = None
        self.__cur_mode = SMALL

        if color == 'B' or color == 'b':
            self.__color = 'blue'  # 目标颜色：红or蓝
        elif color == 'R' or color == 'r':
            self.__color = 'red'

        self.__rect = None  # 目标装甲板矩形
        self.__cur_point = (0, 0)  # 目标实际点
        self.__pre_point = (0, 0)  # 目标预测点
        self.__distance = 0.0  # 实际距离，单位m
        self.__STATUS = 0  # 状态指示--> not_found(0), found(1), fitting(2)
        self.__box = []  # 装甲板矩形
        self.__pre_box = []  # 预测的装甲板矩形
        self.__w_scale = 1
        self.__h_scale = 1

        self.__center_change = 0  # 圆心是否变化，0：未改变 / 1：改变
        self.__fan_change = 0  # 待击打扇叶是否变化，0：未改变 / 1：改变

    def mark(self, frame, time_left=None):

        start_time = time.time()
        # mark之前先初始化
        self.__frame = frame
        self.__rect = None  # 目标装甲板矩形
        self.__cur_point = (0, 0)  # 目标实际点
        self.__pre_point = (0, 0)  # 目标预测点
        self.__distance = 0.0  # 实际距离，单位m
        self.__STATUS = 0  # 状态指示--> found(1) and not_found(0)
        self.__box = []  # 装甲板矩形
        self.__pre_box = []
        self.__w_scale = 1
        self.__h_scale = 1
        self.__center_change = 0
        self.__fan_change = 0

        # ----------- 图像预处理 start -----------
        img = self.__frame.copy()

        if USE_HSV:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
            color_dict = color_list()

            lower = color_dict[self.__color][0]
            upper = color_dict[self.__color][1]
            mask = cv2.inRange(cvt_img, lower, upper)

            if self.__color == 'red':
                lower_r2 = color_dict['red2'][0]
                upper_r2 = color_dict['red2'][1]
                mask_r2 = cv2.inRange(cvt_img, lower_r2, upper_r2)
                mask = mask + mask_r2
        else:
            cvt_img = cv2.cvtColor(img, cv2.COLOR_BGR2YUV)
            if self.__color == 'red':
                mask = cv2.inRange(cvt_img, R_YUV_LOW, R_YUV_HIGH)
            elif self.__color == 'blue':
                mask = cv2.inRange(cvt_img, B_YUV_LOW, B_YUV_HIGH)

        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (CORE_SIZE, CORE_SIZE))
        # 
        mask = cv2.dilate(mask, kernel)
        # mask = cv2.erode(mask, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel)
        mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel)
        # mask = cv2.dilate(mask, kernel)
        # mask = cv2.morphologyEx(mask, cv2.MORPH_ERODE, kernel)
        # ----------- 图像预处理 end ------------

        if self.__debug:
            cv2.namedWindow('mask', 0)
            cv2.resizeWindow('mask', int(1200 * (800 - 80) / 800), 800 - 80)
            cv2.imshow('mask', mask)

        # 获取轮廓
        mask_cp = mask.copy()
        cnts, hierarchy = cv2.findContours(mask_cp, cv2.RETR_TREE, cv2.CHAIN_APPROX_SIMPLE)

        parent_edge = {}
        R_edge = []
        targets = []

        ###############
        if self.__debug:
            orig1 = img.copy()
            for i, c in enumerate(cnts):
                if cv2.contourArea(c) < MIN_AREA:
                    continue
                rect = cv2.minAreaRect(c)
                box = cv2.boxPoints(rect)
                box = np.int0(box)
                orig1 = cv2.drawContours(orig1, [box], 0, (0, 255, 0), 3)
                orig1 = cv2.putText(orig1, str(i), tuple(box[0]), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.namedWindow('contours', 0)
            cv2.resizeWindow('contours', int(1200 * (800 - 80) / 800), 800 - 80)
            cv2.imshow('contours', orig1)
        ###############

        # ----------- 按约束条件筛选轮廓 start -----------
        for i, c in enumerate(cnts):
            if cv2.contourArea(c) < MIN_AREA:
                continue
            if hierarchy[0][i][3] == -1:
                parent_edge[i] = []
            else:
                try:
                    parent_edge[hierarchy[0][i][3]].append(i)
                except KeyError:
                    pass

        for key, value in parent_edge.items():
            rect = cv2.minAreaRect(cnts[key])
            length = max(rect[1][0], rect[1][1])
            width = min(rect[1][0], rect[1][1])

            # 只有一个子轮廓的父轮廓才是目标装甲板
            if len(value) == 1:
                # 4个约束条件：父轮廓外接矩形的长宽比 & 轮廓占外接矩形的比例 & 目标装甲板占外接矩形的比例 & 目标装甲板的长宽比
                rect_armor = cv2.minAreaRect(cnts[value[0]])
                w_armor = max(rect_armor[1][0], rect_armor[1][1])
                h_armor = min(rect_armor[1][0], rect_armor[1][1])
                area_ratio = (w_armor * h_armor) / (length * width)

                if self.__debug:
                    pass
                    # print(length / width, cv2.contourArea(cnts[key]) / (length * width), area_ratio)
                if ASPECT_RATIO[0] < length / width < ASPECT_RATIO[1] \
                        and AREA_LW_RATIO[0] < cv2.contourArea(cnts[key]) / (length * width) < AREA_LW_RATIO[1] \
                        and ARMOR_AREA_RATIO[0] < area_ratio < ARMOR_AREA_RATIO[1]:
                    if ARMOR_WH_RATIO[0] < w_armor / h_armor < ARMOR_WH_RATIO[1]:
                        self.__h_scale = TARGET_HEIGHT / (FAN_LEN * (h_armor / length))
                        self.__w_scale = 1 / (w_armor / width)
                        targets.append((key, value[0], rect_armor))

                # if ASPECT_RATIO_1[0] < length / width < ASPECT_RATIO_1[1] \
                #         and AREA_LW_RATIO_1[0] < cv2.contourArea(cnts[key]) / (length * width) < AREA_LW_RATIO_1[1] \
                #         and ARMOR_AREA_RATIO_1[0] < area_ratio < ARMOR_AREA_RATIO_1[1]:
                #
                #     if ARMOR_WH_RATIO[0] < w_armor / h_armor < ARMOR_WH_RATIO[1]:
                #         self.__h_scale = TARGET_HEIGHT / (TRUE_RADIUS * (h_armor / length))
                #         self.__w_scale = 1 / (w_armor / width)
                #         targets.append((key, value[0], rect_armor))

            # 没有子轮廓的父轮廓可能是R标
            elif len(value) == 0 or len(value) == 1:
                # 限制R标的长宽比
                if R_ASPECT_RATIO[0] < length / width < R_ASPECT_RATIO[1]:
                    R_edge.append((key, rect))
        # ----------- 按约束条件筛选轮廓 end ------------

        if len(targets) == 1:
            self.__STATUS = 1
            # 满足条件的装甲板
            rect_armor = targets[0][-1]
            self.__rect = rect_armor
            self.__box = np.int0(cv2.boxPoints(rect_armor))
            self.__cur_point = cur_point = tuple(np.int0(rect_armor[0]))

            if calc_distance(cur_point, self.last_point) >= MIN_DIS:
                self.__fan_change = 1

            if self.__STATUS == 1:
                # ----------- 寻找圆心的位置 start -----------
                if cur_point:
                    rect_parent = cv2.minAreaRect(cnts[targets[0][0]])
                    if not self.__get_r_method:
                        self.radius, self.r_center = get_r_by_position(rect_armor, rect_parent, R_edge, cur_point)
                    else:
                        # 拟合圆
                        if self.last_point:
                            if self.__fan_change == 1:
                                # 目标点变化较大，重新拟合圆
                                self.fit_circle_points = []
                            left_mid, right_mid = get_mid_pnts(self.__rect, self.__box)
                            if left_mid is not None:
                                self.fit_circle_points.append(tuple(left_mid))
                            self.fit_circle_points.append(self.__cur_point)
                            if right_mid is not None:
                                self.fit_circle_points.append(tuple(right_mid))
                            if len(self.fit_circle_points) > FIT_C_LEN:
                                for _ in range(len(self.fit_circle_points) - FIT_C_LEN):
                                    self.fit_circle_points.pop(0)
                            elif len(self.fit_circle_points) == FIT_C_LEN:
                                # print(self.last_points)
                                self.radius, self.r_center = get_r_by_circle(self.fit_circle_points)

                    if not (self.last_r_center and self.r_center):
                        pass
                    else:
                        if calc_distance(self.r_center, self.last_r_center) < MIN_R_DIFF:
                            self.r_center = self.last_r_center
                        else:
                            self.__center_change = 1

                    self.last_r_center = self.r_center
                # ----------- 寻找圆心的位置 end -------------

                # 测距
                rect_width = max(rect_armor[1][0], rect_armor[1][1])
                dis = get_distance(rect_width, self.__w_scale)
                self.__distance = float(dis[0:-1])

                # 测速 & 大小符判断 & 预测
                if USE_PREDICT:
                    if self.r_center:
                        if self.last_calc_point is None:
                            self.last_calc_point = cur_point

                        if time_left in range(60, 175):
                            # 大符
                            self.__cur_mode = BIG
                            if self.big_start_time:
                                if time.time() - self.big_start_time >= REFIT_THRESH:
                                    self.big_start_time = 0
                                    self.fit_curve_success = False
                            if time.time() - self.last_time >= INTERVAL:
                                # 每隔0.2s计算转速
                                if self.last_point:
                                    # 当前点与上一帧中的点转过的角度
                                    rot_angle = \
                                        calc_angle(cur_point, self.last_calc_point, self.r_center)
                                    # rot_angle = abs(rect_armor[-1] - self.last_angle)
                                    rot_spd = 0.
                                    interval = time.time() - self.last_time

                                    if self.__fan_change == 1 or self.__center_change == 1:
                                        # 目标扇叶发生跳变 or 圆心位置发生变化
                                        rot_spd = self.last_spd
                                    else:
                                        rot_spd = (rot_angle / interval) / 6  # 转速 RPM
                                        if rot_spd < SPD_RANGE[0] or rot_spd > SPD_RANGE[1]:
                                            # 出现检测异常
                                            rot_spd = self.last_spd

                                    self.last_time = time.time()
                                    self.last_calc_point = cur_point

                                    if not self.fit_curve_success:
                                        # 标志位为0 或者 据上次拟合成功的时间超出阈值
                                        if len(self.speeds) < LAST_SPD_LEN:
                                            if len(self.speeds) <= 5:
                                                self.jump_flag = True
                                            else:
                                                self.jump_flag = False
                                            self.speeds.append(rot_spd)
                                            self.intervals.append(interval)
                                        else:
                                            self.speeds.pop(0)
                                            self.speeds.append(rot_spd)

                                            self.intervals.pop(0)
                                            self.intervals.append(interval)

                                            if not self.jump_flag:
                                                if self.__debug:
                                                    print(self.intervals, ',', self.speeds)
                                                popt, pcov = fit_curve(self.intervals, self.speeds)
                                                err = np.sqrt(np.diag(pcov))
                                                print(err)
                                                print('-------------------------\n')

                                                tmp_flag = 0
                                                for e in err:
                                                    if e >= 0.1:
                                                        tmp_flag = 1
                                                        break
                                                if tmp_flag:
                                                    self.fit_curve_success = False
                                                else:
                                                    self.fit_curve_success = True
                                                    self.big_start_time = self.last_time - sum(self.intervals)
                                                    self.popt = popt

                                            if time.time() - self.last_detec_time >= 2:
                                                self.last_detec_time = time.time()
                                    else:
                                        popt = self.popt
                                        rot_spd = spd_func(time.time() - self.big_start_time,
                                                           popt[0], popt[1], popt[2], popt[3])

                                    self.rot_spd = rot_spd
                                    self.last_spd = self.rot_spd
                                    # print(round(rot_spd, 2), self.rot_spd, self.__cur_mode)
                        else:
                            # 小符
                            self.__cur_mode = SMALL
                            self.rot_spd = 16

                        # 转向判断
                        if len(self.rot_direction) == 0 or time.time() - self.last_detec_time >= 2:
                            # 每两秒检测一下转动方向
                            if self.__center_change == 0 and len(self.last_points) >= LAST_P_LEN:
                                for i in range(LAST_P_LEN):
                                    tmp = [self.last_points[i], cur_point]
                                    section = get_section(self.r_center, tmp[0], tmp[1])
                                    if section != -1:
                                        # self.rot_direction.append(1)
                                        if len(self.rot_direction) == 0:
                                            self.rot_direction.append(get_rotate_direction(tmp, section))
                                        else:
                                            self.rot_direction[0] = get_rotate_direction(tmp, section)
                                        break
                        # 预测点与ROI
                        if self.rot_direction:
                            if self.rot_direction[0] == 0:
                                theta = self.getTheta()
                            else:
                                theta = -1 * self.getTheta()
                            rot_mat = cv2.getRotationMatrix2D(self.r_center, theta, 1)
                            sinA = rot_mat[0][1]
                            cosA = rot_mat[0][0]

                            self.__pre_point = self.getRotPoint(sinA, cosA, cur_point)
                            for p in list(self.__box):
                                self.__pre_box.append(
                                    self.getRotPoint(sinA, cosA, p)
                                )
                            self.__pre_box = np.array(self.__pre_box)

                if not USE_PREDICT \
                        or not self.r_center \
                        or calc_distance(self.__pre_point, cur_point) > MIN_DIS_PRED:
                    self.__pre_point = cur_point
                    self.__pre_box = self.__box

                self.last_point = cur_point
                if self.__fan_change:
                    self.last_points = []
                else:
                    if len(self.last_points) < LAST_P_LEN:
                        self.last_points.append(cur_point)
                    else:
                        self.last_points.pop(0)
                        self.last_points.append(cur_point)
        else:
            # 此时该帧图像中没有满足要求的装甲板 或者 有过多满足要求的装甲板
            self.__STATUS = 0
            pass

    def markFrame(self):
        """
        画图函数
        :return:一帧标记好的图像
        """
        orig = self.__frame.copy()
        try:
            pre_box = self.__pre_box
            pre_box = get_vertex(self.__pre_box, cv2.minAreaRect(self.__pre_box))
            for p in self.fit_circle_points:
                orig = cv2.circle(orig, (int(p[0]), int(p[1])), 3, (0, 255, 0), 1)

            # for i in range(len(pre_box)):
            #     orig = cv2.circle(orig, (pre_box[i][0], pre_box[i][1]), 5, (0, 255, 0), 2)
            #     orig = cv2.putText(orig, str(i), (pre_box[i][0] + 10, pre_box[i][1] - 2), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
            #                        (0, 0, 255), 1)
            orig = cv2.drawContours(orig, [pre_box], 0, (0, 255, 255), 3)
            orig = cv2.putText(orig, str(round(self.rot_spd, 2)), self.__cur_point, cv2.FONT_HERSHEY_SIMPLEX, 1,
                               (0, 255, 0), 2)
            if self.r_center != (0, 0):
                orig = cv2.circle(orig, (int(self.r_center[0]), int(self.r_center[1])), 5, (0, 255, 0), 3)
            # orig = cv2.circle(orig, (int(self.__cur_point[0]), int(self.__cur_point[1])), self.__cur_point,
            # (0, 255, 0), 3)
            # orig = cv2.circle(orig, self.__pre_point, 10, (0, 255, 0), 3)
        except:
            pass
        if self.__STATUS == 0:
            s = 'not found'
        elif self.__STATUS == 1:
            s = 'found'
        elif self.__STATUS == 2:
            s = 'fitting'
        else:
            s = 'not found'
        orig = cv2.putText(orig, self.__color + ' ' + s, (30, 30), cv2.FONT_HERSHEY_SIMPLEX, 1,
                           (0, 255, 0), 2)

        return orig

    def getResult(self):
        """
        返回值说明：[status, image_points, model_points]
            - status：      0\1\2，指示大符程序目前状态，0->'not found', 1->'found'，2->'fitting'
            - image_points:     预测点在图像中的坐标
            - model_points:     以装甲板中心为原点，装甲板四个角点的坐标
        """

        if self.__STATUS and len(self.__pre_box) != 0:
            # pnp解算
            model_points = get_model(self.__w_scale, self.__h_scale)
            pre_box = get_vertex(self.__pre_box, cv2.minAreaRect(self.__pre_box))
            # 中点、左下、左上、右上、右下
            image_points = np.concatenate([np.array([self.__pre_point]), pre_box], axis=0)
            image_points = np.array(image_points, dtype=np.float)
            return [self.__STATUS, image_points, model_points]

            # 小孔成像解算
            # size = self.__frame.shape
            # position = getPosition(self.__pre_point, (size[1] / 2, size[0] / 2), self.__distance,
            #                        self.__distance / FOCAL_LENGTH)
            # return [self.__STATUS, position]
        else:
            return [self.__STATUS]

    def getTheta(self):
        """
        返回下一个点与当前点的角度
        """
        return ((self.__distance / self.__init_v) * self.rot_spd * np.pi / 30) * (180 / np.pi)

    def getRotPoint(self, sinA, cosA, p):
        """
        p点绕r_center 旋转后的点
        :param sinA:
        :param cosA:
        :param p:
        :return: 旋转后点的坐标(x,y)
        """
        xx = -(self.r_center[0] - p[0])
        yy = -(self.r_center[1] - p[1])

        return (
            int(self.r_center[0] + cosA * xx - sinA * yy),
            int(self.r_center[1] + sinA * xx + cosA * yy))
