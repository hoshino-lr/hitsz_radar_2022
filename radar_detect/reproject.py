"""
反投影预警
created by 牟俊宇 2020/12
最新修改 by 牟俊宇 2021/1/15
"""
from typing import Union, Any, Tuple
from numpy import ndarray

import cv2
import numpy as np
import time

from radar_detect.common import is_inside
from resources.config import color2enemy, enemy_case, enemy_color, cam_config, \
    DEBUG, test_region, region, real_size


class Reproject(object):
    """
    反投影预警类
    """
    _iou_threshold = 0.8  # iou阈值
    _clock = 1  # 间隔一秒发送text
    _frame = 3  # 一秒钟需要检测到3帧

    def __init__(self, name, textapi):
        # 读取相机配置文件
        self._tvec = cam_config[name]['tvec']
        self._rvec = cam_config[name]['rvec']
        self._K_O = cam_config[name]['K_0']
        self._C_O = cam_config[name]['C_0']
        if DEBUG:  # 若为debug模式
            self._region = test_region
        else:
            self._region = region
        self._scene_region = {}  # 反投影位置图形
        self._name = name
        self._enemy = enemy_color
        self._real_size = real_size  # 真实尺寸
        self._scene_init = False  # 初始化flag
        self.fly = False
        self._textapi = textapi  # 发送text的api
        self.rp_alarming = {}

        self._region_count = {}  # 检测区域计数
        self._time = {}  # 时间间隔
        self._start = {}  # 开始时间
        self._end = {}  # 结束时间

        self._plot_regin()  # 预警区域坐标初始化

    def _plot_regin(self) -> None:
        """
        计算预警区域反投影坐标，用第一帧初始化了就可以了。
        """
        for r in self._region.keys():
            # 格式解析
            cor = None
            height = None
            rtype, shape_type, team, location, height_type = r.split('_')
            if location not in enemy_case or color2enemy[team] == self._enemy:  # 筛选出敌方的区域
                if rtype == 's' or rtype == 'a':  # 筛选需要进行反投影的区域
                    self._region_count[f'{location}'] = 0  # 初始化区域检测计数
                    self._time[f'{location}'] = 0  # 初始化时间间隔
                    if shape_type == 'r':  # 计算矩形坐标
                        # 引入左上右下两点
                        lt = self._region[r][:2].copy()
                        rd = self._region[r][2:4].copy()
                        # 另外两点坐标
                        ld = [lt[0], rd[1]]
                        rt = [rd[0], lt[1]]
                        cor = np.float32([lt, rt, rd, ld]).reshape(-1, 2)  # 坐标数组
                        if height_type == 'a':  # 四点在同一高度
                            height = np.ones((cor.shape[0], 1)) * self._region[r][4]
                        if height_type == 'd':  # 四点在不同高度
                            height = np.ones((cor.shape[0], 1))
                            height[1:3] *= self._region[r][5]  # 右上和右下
                            height[[0, 3]] *= self._region[r][4]  # 左上和左下
                    if shape_type == 'fp':
                        # 四点凸四边形类型，原理同上
                        cor = np.float32(self._region[r][:8]).reshape(-1, 2)
                        # cor[:, 1] -= self._real_size[1]  # 同上，原点变换
                        if height_type == 'a':
                            height = np.ones((cor.shape[0], 1)) * self._region[r][8]
                        if height_type == 'd':
                            height = np.ones((cor.shape[0], 1))
                            height[1:3] *= self._region[r][9]
                            height[[0, 3]] *= self._region[r][8]
                    if isinstance(cor, np.ndarray) and isinstance(height, np.ndarray):
                        cor = np.concatenate([cor, height], axis=1) * 1000  # 合并高度坐标 顺便把米转换为毫米
                        recor = cv2.projectPoints(cor, self._rvec, self._tvec, self._K_O, self._C_O)[0] \
                            .astype(int).reshape(-1, 2)  # 得到反投影坐标
                        self._scene_region[r] = recor  # 储存反投影坐标
                        self._scene_init = True

    def push_T(self, rvec: ndarray, tvec: ndarray) -> None:
        """
        输入相机位姿（世界到相机）
        """
        self._rvec = rvec
        self._tvec = tvec
        self._plot_regin()  # 初始化预警区域字典

    def update(self, frame) -> None:
        """
        更新一帧绘图
        """
        for i in self._scene_region.keys():
            recor = self._scene_region[i]
            _, shape_type, team, location, height_type = i.split('_')
            if color2enemy[team] != enemy_color:
                continue
            else:
                if i in self.rp_alarming.keys():
                    cv2.polylines(frame, [recor], isClosed=True, color=(0, 255, 0), thickness=1)
                else:
                    cv2.polylines(frame, [recor], isClosed=True, color=(0, 0, 255), thickness=1)

    def check(self, net_input) -> None:
        """
        预警预测
        Args:
            net_input:输入
        """
        armors = None  # armors:N,cls+对应的车辆预测框序号+装甲板bbox
        cars = None  # cars:N,cls+车辆bbox
        self.fly = False
        self.rp_alarming = {}
        if isinstance(net_input, np.ndarray):  # 解析网络输入
            if len(net_input):
                armors = net_input[:, [11, 13, 6, 7, 8, 9]]
                cars = net_input[:, [11, 0, 1, 2, 3]]

        color_bbox = []
        if isinstance(armors, np.ndarray) and isinstance(cars, np.ndarray):
            cls = armors[:, 0].reshape(-1, 1)
            # 默认使用bounding box为points四点
            x1 = armors[:, 2].reshape(-1, 1)
            y1 = armors[:, 3].reshape(-1, 1)
            x2 = armors[:, 4].reshape(-1, 1)
            y2 = armors[:, 5].reshape(-1, 1)
            points = np.concatenate([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)
            # 对仅预测出颜色的敌方预测框进行数据整合
            for i in cars:
                if i[0] == 0:
                    color_bbox.append(i)
            if len(color_bbox):
                color_bbox = np.stack(color_bbox, axis=0)
            if isinstance(color_bbox, np.ndarray):
                # 预估装甲板位置，见技术报告
                color_cls = color_bbox[:, 0].reshape(-1, 1)
                color_bbox[:, 3] = (color_bbox[:, 3] - color_bbox[:, 1]) // 3
                color_bbox[:, 4] = (color_bbox[:, 4] - color_bbox[:, 2]) // 5
                color_bbox[:, 1] += color_bbox[:, 3]
                color_bbox[:, 2] += color_bbox[:, 4] * 3
                x1 = color_bbox[:, 1]
                y1 = color_bbox[:, 2]
                x2 = x1 + color_bbox[:, 3]
                y2 = y1 + color_bbox[:, 4]
                color_fp = np.stack([x1, y1, x2, y1, x2, y2, x1, y2], axis=1)
                # 与之前的数据进行整合
                points = np.concatenate([points, color_fp], axis=0)
                cls = np.concatenate([cls, color_cls], axis=0)
            points = points.reshape((-1, 4, 2))
            for r in self._scene_region.keys():
                # 判断对于各个预测框，是否有点在该区域内
                mask = np.array([[is_inside(self._scene_region[r], p) for p in cor] for cor in points])
                mask = np.sum(mask, axis=1) > 0  # True or False,只要有一个点在区域内，则为True
                alarm_target = cls[mask]  # 需要预警的装甲板种类
                if len(alarm_target):
                    self.rp_alarming = {r: alarm_target.reshape(-1, 1)}

    def push_text(self) -> None:
        """
        发送信息
        """
        if self._scene_init:
            for r in self.rp_alarming.keys():
                # 格式解析
                _, _, _, location, _ = r.split('_')
                if location == "3号高地":
                    for i in self.rp_alarming[r]:
                        try:
                            if i == 1:
                                self._textapi("ERROR", f"反投影{location}", f"在{location}处有英雄！！！\n")
                                print(f"[ERROR] 反投影{location}", f"在{location}处有英雄！！！")
                        except Exception as e:
                            print(e)
                else:
                    if self._time[f'{location}'] == 0:
                        self._start[f'{location}'] = time.time()
                        self._region_count[f'{location}'] += 1
                        self._end[f'{location}'] = time.time()
                        self._time[f'{location}'] = self._end[f'{location}'] - self._start[f'{location}']
                    else:
                        self._end[f'{location}'] = time.time()
                        self._time[f'{location}'] = self._end[f'{location}'] - self._start[f'{location}']
                        if self._time[f'{location}'] <= 1:
                            self._region_count[f'{location}'] += 1
                        if self._time[f'{location}'] >= self._clock:
                            if self._region_count[f'{location}'] >= self._frame:
                                if location == "飞坡":
                                    self.fly = True
                                self._textapi("ERROR", f"反投影{location}", f"在{location}处有敌人！！！\n")
                                print(f"[ERROR] 反投影{location}", f"在{location}处有英雄！！！")
                                self._region_count[f'{location}'] = 0
                            self._time[f'{location}'] = 0
        else:
            self._textapi("WARNING", "反投影", "未初始化！！！")


if __name__ == "__main__":
    ori = cv2.imread("/home/hoshino/CLionProjects/hitsz_radar/resources/beijing.png")
    frame = ori.copy()
    repo = Reproject(frame=frame, name='cam_left')
    _, _, region11 = repo.push_T(cam_config['cam_left']['rvec'], cam_config['cam_left']['tvec'])

    frame = ori.copy()
    rect_armor = cv2.selectROI("img", frame, False)
    rect_car = cv2.selectROI("img", frame, False)
    key = cv2.waitKey(1)
    while key != ord('q'):
        key = cv2.waitKey(1)
        # 在初始输入上绘制
        for i in region11.keys():
            recor = region11[i]
            for p in recor:
                cv2.circle(frame, tuple(p), 10, (0, 255, 0), -1)
            cv2.fillConvexPoly(frame, recor, (0, 255, 0))
        # 为了统一采用is_inside来判断是否在图像内
        # 分别在实际相机图和深度图上画ROI框来对照
        cv2.rectangle(frame, (rect_car[0], rect_car[1]), (rect_car[0] + rect_car[2]
                                                          , rect_car[1] + rect_car[3]), (0, 255, 0), 3)
        cv2.rectangle(frame, (rect_armor[0], rect_armor[1]), (rect_armor[0] + rect_armor[2]
                                                              , rect_armor[1] + rect_armor[3]), (0, 255, 0), 3)
        cv2.imshow("img", frame)
        if key == ord('r') & 0xFF:
            # 重选区域
            rect_armor = cv2.selectROI("img", frame, False)
        if key == ord('c') & 0xFF:
            # 重选区域
            rect_car = cv2.selectROI("img", frame, False)
        if key == ord('p') & 0xFF:
            # 重选区域
            frame = ori.copy()
        if key == ord('s') & 0xFF:
            # 显示世界坐标系和相机坐标系坐标和深度，以对测距效果进行粗略测试
            armor = np.array([1, 0, rect_armor[0], rect_armor[1], rect_armor[2], rect_armor[3]]).reshape((-1, 6))
            car = np.array([1, rect_car[0], rect_car[1], rect_car[2], rect_car[3]]).reshape((-1, 5))
            result = repo.check(armors=armor, cars=car)
            print(result)
