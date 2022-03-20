"""
飞镖预警
"""
import cv2
import numpy as np
import traceback
import time
from common import is_inside


def missile_filter(frame_m, red=True):
    """
    飞镖滤波

    :param frame_m: 图像
    :param red: 敌方
    """
    if red:
        l = 170
        h = 180
    else:
        l = 95
        h = 100
    intensity_thre = 175  # 亮度阈值
    gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    gray[gray < intensity_thre] = 0
    # HSV threshold 可调参
    frame_hsv = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
    mask = cv2.inRange(frame_hsv, (l, 170, 0), (h, 255, 255))
    # 膨胀
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (7, 7))
    mask = cv2.dilate(mask, kernel)
    gray[mask < 200] = 0
    return gray


class Missile(object):
    """
    飞镖预警类
    """
    _region_thre = [100, 40000]  # 响应区域bounding box大小上下限
    _intensity_bound = 175  # 亮度

    def __init__(self, enemy, debug=False):
        """
        :param enemy: 敌方
        :param debug: 调参
        """
        self._init_flag = False
        self._debug = debug
        self._enemy = enemy
        self._two_stage_time = 0  # 第二阶段开始的时间
        self._region = [None, None]
        self._roi = [None, None]
        self._previous_frame = [None, None]
        self._open = False
        self._y_c = self._y_p = 0
        if debug:
            # 调参用的，只针对第二阶段,可适当修改
            cv2.namedWindow("missile_debug", cv2.WINDOW_NORMAL)
            cv2.createTrackbar("lb", "missile_debug", 0, 1000, lambda x: None)
            cv2.setTrackbarMax("lb", "missile_debug", 1000)
            cv2.setTrackbarMin("lb", "missile_debug", 100)
            cv2.setTrackbarPos("lb", "missile_debug", 500)
            cv2.createTrackbar("s", "missile_debug", 0, 255, lambda x: None)
            cv2.setTrackbarMax("s", "missile_debug", 255)
            cv2.setTrackbarMin("s", "missile_debug", 0)
            cv2.setTrackbarPos("s", "missile_debug", 200)

    def _push_init(self, init_frame, region):
        """
        进行初始化

        :param init_frame : 初始图像
        :param region : 检测区域，需要先进行反投影
        """
        if not self._init_flag:
            # 此处为源代码，并未进行测试
            # for i in range(2):
            #     r = region[f's_fp_{enemy2color[self._enemy]}_missilelaunch{i + 1:d}_d'].copy()  # 凸四边形
            #     rect = cv2.boundingRect(r)
            #     r -= np.array(rect[:2])  # 以外bounding box左上角为原点的凸四边形坐标
            # 存储取图像ROI区域的信息
            # self._roi[i] = rect
            # self._region[i] = r.copy()
            # 存储帧差第一帧
            # self._previous_frame[i] = init_frame[self._roi[i][1]:(self._roi[i][1] + self._roi[i][3]),
            #                       self._roi[i][0]:(self._roi[i][0] + self._roi[i][2])].copy()
            # 以下为自己框选的区域
            self._roi[0] = [700, 910, 1300, 1400]
            self._roi[1] = [300, 580, 1250, 1800]
            self._region[0] = np.array([0, 0, 100, 0, 100, 210, 0, 210]).reshape((4, 2))
            self._region[1] = np.array([0, 0, 550, 0, 550, 280, 0, 280]).reshape((4, 2))
            self._previous_frame[0] = init_frame[700:910, 1300:1400].copy()
            self._previous_frame[1] = init_frame[300:580, 1250:1800].copy()
            self._init_flag = True

    def detect(self, img: np.ndarray, region: dict, stage):
        """
        :param img: 每一帧
        :param region: 检测区域
        :param stage: 第几阶段
        """
        if self._debug:
            self._region_thre[0] = cv2.getTrackbarPos("lb", "missile_debug")
            self._intensity_bound = cv2.getTrackbarPos("s", "missile_debug")
        try:
            # 若未初始化
            if not self._init_flag:
                self._push_init(img, region)
                return False
            else:
                detect_flag = False
                kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
                # 取对应阶段当前帧ROI区域
                current_frame = img[self._roi[stage][0]:self._roi[stage][1],
                                self._roi[stage][2]:self._roi[stage][3]].copy()
                if stage == 0:
                    # 第一阶段处理，亮度二值化+帧差
                    current_frame_gray = cv2.cvtColor(current_frame, cv2.COLOR_BGR2GRAY)
                    previous_frame_gray = cv2.cvtColor(self._previous_frame[stage], cv2.COLOR_BGR2GRAY)
                    current_frame_gray = cv2.GaussianBlur(current_frame_gray, (7, 7), 0)
                    previous_frame_gray = cv2.GaussianBlur(previous_frame_gray, (7, 7), 0)
                    current_frame_gray[current_frame_gray < self._intensity_bound] = 0
                    previous_frame_gray[previous_frame_gray < self._intensity_bound] = 0
                    # 计算轮廓
                    contours_p, _ = cv2.findContours(previous_frame_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours_p:
                        if self._region_thre[0] < cv2.contourArea(c) < self._region_thre[1]:
                            x, y, w, h = cv2.boundingRect(c)
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                self._y_p = y
                    contours_c, _ = cv2.findContours(current_frame_gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours_c:
                        if self._region_thre[0] < cv2.contourArea(c) < self._region_thre[1]:
                            x, y, w, h = cv2.boundingRect(c)
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                self._y_c = y
                    frame_diff = cv2.absdiff(current_frame_gray, previous_frame_gray)  # 进行帧差
                    _, frame_diff = cv2.threshold(frame_diff, 10, 255, cv2.THRESH_BINARY)
                    frame_diff = cv2.erode(frame_diff, kernel)
                    frame_diff = cv2.dilate(frame_diff, kernel)
                    contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if self._region_thre[0] < cv2.contourArea(c) < self._region_thre[1]:
                            x, y, w, h = cv2.boundingRect(c)
                            # 中心点是否在凸四边形区域内
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                detect_flag = True
                                self._open = True
                                if self._debug:
                                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255))
                                else:
                                    break
                        else:
                            if self._open and self._y_c < self._y_p:
                                print("已经完全打开")
                                self._open = False

                else:
                    # 第二阶段处理，飞镖滤波函数（亮度+HSV区域二值化）+帧差
                    current_frame_filter = missile_filter(current_frame,
                                                          not self._enemy)  # not self._enemy 来选择是进行红滤波还是蓝滤波
                    previous_frame_filter = missile_filter(self._previous_frame[stage], not self._enemy)
                    for i in range(2):
                        self._previous_frame[i] = img[self._roi[i][0]:self._roi[i][1],
                                                  self._roi[i][2]:self._roi[i][3]].copy()
                    frame_diff = cv2.absdiff(current_frame_filter, previous_frame_filter)
                    frame_diff = cv2.dilate(frame_diff, kernel)
                    contours, _ = cv2.findContours(frame_diff, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
                    for c in contours:
                        if cv2.contourArea(c) > 30:  # 轮廓面积阈值
                            x, y, w, h = cv2.boundingRect(c)
                            flag = is_inside(self._region[stage], np.array([x + w // 2, y + h // 2]))
                            if flag:
                                print("Second stage detect")
                                detect_flag = True
                                if self._debug:
                                    cv2.rectangle(current_frame, (x, y), (x + w, y + h), (0, 0, 255))
                                else:
                                    break

                if self._debug:
                    cv2.imshow('missile_debug', current_frame)
                cv2.imshow("c", current_frame)
                cv2.imshow("p", self._previous_frame[stage])
                for i in range(2):
                    # 储存为上一帧
                    self._previous_frame[i] = img[self._roi[i][0]:self._roi[i][1],
                                              self._roi[i][2]:self._roi[i][3]].copy()
                return detect_flag
        except Exception:
            traceback.print_exc()
            return False

    def init_stage2(self):
        """
        云台手按下按钮后，启动计时
        """
        self._two_stage_time = time.time()

    def detect_stage2(self, img, region=None):
        """
        开启第二阶段检测
        :param img: 检测图片
        :param region: 检测区域
        :return: 是否正在检测，是否检测到发射
        """
        launch = False
        detect_flag = self.detect(img, region, 1)
        if time.time() - self._two_stage_time > 20.:  # 若处于反导侦测阶段TWO_STAGE_TIME秒，则自动结束
            print("missile end")
            return False, False
        if detect_flag:
            self._touch_api({"task": 4})
            launch = True
        return True, launch


if __name__ == '__main__':
    region = {'1': 0}
    t = Missile(0)
    cap = cv2.VideoCapture()
    cap.open("1.mp4")
    while cap.isOpened():
        ret, frame = cap.read()
        if ret:

            t.detect(frame, region, 0)

        else:
            print("视频播放完成！")
            break

        # 退出播放
        key = cv2.waitKey(30)
        if key == 27:  # 按键esc
            break

        # 3.释放资源
    cap.release()
    cv2.destroyAllWindows()
