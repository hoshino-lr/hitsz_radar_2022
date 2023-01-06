"""
位置预警类，还需要修改和完善
created by 黄继凡 2021/12
最新修改 by 黄继凡 2022/7/10
"""
import pickle

import cv2 as cv
import numpy as np
import time
import pickle as pkl
from radar_detect.common import is_inside_polygon
import mapping.draw_map as draw_map  # 引入draw_map模块，使用其中的CompeteMap类
from config import armor_list, color2enemy, enemy_color, cam_config, real_size, region, test_region, choose
from radar_detect.location_Delaunay import location_Delaunay


class Alarm(draw_map.CompeteMap):
    """
    预警类，继承自地图画图类
    删除原先的refresh、show函数，
    two_camera_merge_update、update函数更改传入参数、
    返回车辆位置字典_location，统一绘图
    """
    # param
    _pred_time = 5  # 预测几次  一开始是10
    _pred_ratio = 0  # 预测速度比例

    con_thre = 0.5  # 置信度阈值
    con_decre1 = 0.075  # 置信度没超过阈值时的衰减速度
    con_incre = 0.10  # 检测到装甲板时，置信度增加速度
    con_decre2 = 0.025  # 置信度超过阈值时的衰减速度

    _lp = True  # 是否位置预测
    _z_a = False  # 是否进行z轴突变调整
    _z_thre = 0.2  # z轴突变调整
    _ground_thre = 100  # 地面阈值，我们最后调到了100就是没用这个阈值，看情况调

    state_name = ['雷达点云定位', '德劳内三角定位', 'kd_tree定位', '禁用定位']
    state = [3, 3]

    def __init__(self, api, touch_api, enemy, state_: list, _save_data: bool, debug=False):
        """
        :param api:主程序显示api，传入画图程序进行调用（不跨线程使用,特别是Qt）
        :param touch_api:log api
        :param enemy:敌方编号
        :param debug:debug模式
        """
        self._debug = debug
        self._save_data = _save_data
        # 敌人颜色
        self._enemy = enemy
        # 显示api
        self._touch_api = touch_api
        self.state = state_
        if debug:
            super(Alarm, self).__init__(test_region, real_size, enemy, api)
            self._region = test_region
        else:
            super(Alarm, self).__init__(region, real_size, enemy, api)
            self._region = region

        self.reset_thre = 300
        self.reset_count = 0

        # current_time 车辆出现时间字典，键为字符'1'-'5'，值为车辆出现的时间点
        self._current_time = {}

        # 下面两个字典的value应该是数组[x, y, z, time]
        # location 车辆位置字典，键为字符'1'-'5'，值为车辆的位置数组，包含车辆坐标及出现时间间隔:[x, y, z, time]
        self._location = {}
        # location 车辆最后一次位置字典，键为字符'1'-'5'，值为车辆的位置数组，包含车辆坐标及出现时间间隔:[x, y, z, time]
        self._last_location = {}
        # confidence 车辆置信度字典，键为字符'1'-'5'，值为车辆的置信度
        self._confidence = {}

        # 一阶低通滤波
        self.filter = 0.5

        # 分别为左右相机存储位置信息，z坐标缓存，相机世界坐标系位置，以及（相机到世界）转移矩阵
        self._locations = [None, None]
        # self._z_cache = [None, None]
        self._camera_position = [None, None]
        self._T = [None, None]
        # 根据敌方颜色确定识别颜色
        if int(enemy_color) == 0:
            choose_left = "cam_left_red"
            choose_right = "cam_right_red"
        else:
            choose_left = "cam_left_blue"
            choose_right = "cam_right_blue"

        # 左右相机的德劳内三角定位
        self._loc_D = [location_Delaunay(
            "cam_left", debug, choose[choose_left]), location_Delaunay("cam_right", debug, choose[choose_right])]

        # 相机内参
        self._K_O = cam_config['cam_left']['K_0']
        self._location_pred_time = np.zeros(5, dtype=int)  # 预测次数记录

        # 对错误值进行预测
        self.thre_predict = [4, 24]

        # 判断x各行是否为全零的函数
        self._f_equal_zero = lambda x: np.isclose(
            np.sum(x, axis=1), np.zeros(x.shape[0]))

        start_time = time.time()
        for i in range(1, 6):  # 初始化位置为全零
            self._location[str(i)] = [0, 0, 0, 0]  # x，y，z, t
            self._current_time[str(i)] = start_time
            self._confidence[i] = 0

        # 初始化定位信息
        self._locations = [np.zeros((5, 3)), np.zeros((5, 3))]
        # 初始化最后一次位置
        self._location_cache = self._location.copy()
        self._last_location = self._location.copy()

        if self._save_data:
            self._f = open("resources/location_data.dat", "wb")

    def close_data(self):
        if self._save_data:
            self._f.close()

    def push_T(self, rvec, tvec, camera_type):
        """
        位姿信息，求算从图像坐标转为世界坐标的矩阵
        :param rvec:旋转矩阵
        :param tvec:平移矩阵
        :param camera_type:相机编号，若为单相机填0
        """
        self._loc_D[camera_type].push_T(rvec, tvec)
        T = np.eye(4)
        T[:3, :3] = cv.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
        T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
        T = np.linalg.inv(T)  # 矩阵求逆
        # 相机的世界坐标
        camera_position = (T @ (np.array([0, 0, 0, 1])))[:3]
        self._camera_position[camera_type] = camera_position.copy()
        self._T[camera_type] = T.copy()

    def change_mode(self, camera_type):
        # 循环切换到下一个定位模式
        self.state[camera_type] = (self.state[camera_type] + 1) % 4

    def _check_alarm(self):
        """
        预警检测
        alarming:各区域是否有预警;
        base_alarming:基地是否有预警
        """
        for loc in self._region.keys():
            alarm_type, team, target, l_type = loc.split('_')
            if color2enemy[team] == self._enemy:
                targets = []
                # 对所有装甲板和所有区域，检测装甲板是否在区域内
                for armor in list(self._location.keys())[0:5]:
                    l_ = np.float32(self._location[armor])
                    # 在区域内
                    if is_inside_polygon(np.array(self._region[loc])[:, :2], l_[0:2]):
                        targets.append(armor)
                    # if alarm_type == 'm' or alarm_type == 'a':  # 若为位置预警
                    #     if shape_type == 'r' and (
                    #             target in enemy_case or color2enemy[team] == self._enemy):  # 对于特殊地点，只考虑对敌方进行预警
                    #         # 矩形区域采用范围判断
                    #         if self._region[loc][0] >= l_[0] >= self._region[loc][2] \
                    #                 and self._region[loc][3] <= l_[1] <= self._region[loc][1]:
                    #             targets.append(armor)
                    #     # base alarm
                    #     if shape_type == 'l' and color2enemy[team] != self._enemy:
                    #         # 直线检测
                    #         up_p = np.float32(self._region[loc][:2])  # 上端点
                    #         dw_p = np.float32(self._region[loc][2:4])  # 下端点
                    #         dis_thres = self._region[loc][4]
                    #         up_l = up_p - dw_p  # 直线向上的向量
                    #         dw_l = dw_p - up_p  # 直线向下的向量
                    #         m_r = np.array([up_l[1], -up_l[0]], dtype=np.float32)  # 方向向量，向右
                    #         m_l = np.array([-up_l[1], up_l[0]], dtype=np.float32)  # 方向向量，向左
                    #
                    #         def f_dis(m):
                    #             return m @ (l_[0:2] - dw_p) / \
                    #                    np.linalg.norm(m)  # 计算从下端点到物体点在各方向向量上的投影
                    #
                    #         if l_type == 'l':
                    #             dis = f_dis(m_l)
                    #         elif l_type == 'r':
                    #             dis = f_dis(m_r)
                    #         else:  # l_type == 'a'
                    #             dis = abs(f_dis(m_r))  # 绝对距离
                    #         # 当物体位置在线段内侧，且距离小于阈值时，预警
                    #         if up_l @ (l_[0:2] - dw_p) > 0 and dw_l @ (l_[0:2] - up_p) > 0 and \
                    #                 dis_thres >= dis >= 0:
                    #             targets.append(armor)
                    #     if shape_type == 'fp' and (target not in enemy_case or color2enemy[team] == self._enemy):
                    #         # 判断是否在凸四边形内
                    #         if is_inside(np.float32(self._region[loc][:8]).reshape(4, 2), point=l_[0:2]):
                    #             targets.append(armor)

                if len(targets):
                    # 发送预警
                    self._add_twinkle(loc)
                    # targets_text = ' '.join(targets)
                    # str_text += f"{target}出现{team} {targets_text}\n"

        # message_ = draw_message("位置预警-信息输出", 0, str_text, "WARNING")
        # self._touch_api(message_)

    def _adjust_z(self, camera_type):
        """
        z轴突变调整，仅针对一个装甲板
        :param camera_type:相机编号
        """
        position = self._T[camera_type][:, 3].copy().reshape(-1)
        t = time.time()
        for i in range(1, 6):
            # z坐标大，则需警惕误识别
            if self._location[str(i)][2] >= 1.2 and (t - self._last_location[str(i)][3]) < 1:
                # 相似三角形 投影到原z坐标对应平面
                z_ratio = (self._location[str(i)][2] - self._last_location[str(i)][2]) / (position[2] - self._location[str(i)][2])
                self._location[str(i)][2] = self._last_location[str(i)][2]
                self._location[str(i)][0] = self._location[str(i)][0] - z_ratio * (
                        position[0] - self._location[str(i)][0])
                self._location[str(i)][1] = self._location[str(i)][1] - z_ratio * (
                        position[1] - self._location[str(i)][1])

    def _location_prediction(self):
        """
        位置预测，更新_last_location
        """
        # 上两帧位置 (2,N)
        # pre = np.float32(list(i[0:3] for i in list(self._location_cache.values())))
        # 该帧预测位置；将包含时间的信息中的位置信息提取出
        # now = np.float32(list(i[0:3] for i in list(self._location.values())))

        # pre1_zero = self._f_equal_zero(pre)  # the last frame 上一帧
        # now_zero = self._f_equal_zero(now)  # the latest frame 当前帧

        # 仅对该帧全零，上帧不为0的id做预测
        # do_prediction = np.logical_and(np.logical_not(pre1_zero), now_zero)

        # now[do_prediction] = pre[do_prediction]

        # z轴突变调整
        if self._z_a:
            self._adjust_z(0)
        for i in range(1, 6):
            # 置信度达到阈值
            if self._confidence[i] >= self.con_thre and self._location[str(i)][0] > 0:
                if self._last_location[str(i)][0] > 0:
                    if self._current_time[str(i)] - self._last_location[str(i)][3] < 1:
                        # 位置变化大，取之前位置
                        if abs(self._last_location[str(i)][0] - self._location[str(i)][0]) + \
                                abs(self._last_location[str(i)][1] - self._location[str(i)][1]) > 3:
                            temp = self._location[str(i)][0:3]
                        else:
                            # 滤波，防止位置抖动
                            temp = (np.array(self._location[str(i)][0:3]) * self.filter + np.array(
                                self._last_location[str(i)][0:3]) * (1 - self.filter)).tolist()
                    else:
                        temp = self._location[str(i)][0:3]
                else:
                    temp = self._location[str(i)][0:3]
                temp.append(self._current_time[str(i)])
                # 更新最新一次的位置信息
                self._last_location[str(i)] = temp
            else:
                self._location[str(i)] = [0, 0, 0, 0]   # 置信度不满足要求，重置位置信息，不再参与预测
            # push new data
            self._location_cache = self._location.copy()

    def check(self):
        """
        预警检测
        """
        self._check_alarm()

    def two_camera_merge_update(self, t_locations_left, t_locations_right, rp_alarming: dict):
        """
        对两个相机的检测结果的合并处理
        :param t_locations_left: 左相机检测到的装甲板位置信息 [N, cls+x+y+z]
        :param t_locations_right: 右相机检测到的装甲板位置信息（没有使用）
        :param rp_alarming: 反投影
        """
        if self._save_data:
            pickle.dump([t_locations_left, t_locations_right, rp_alarming], self._f)
        # init location
        for i in range(1, 6):
            self._location[str(i)][0:2] = [0, 0]
        # 更新左右相机检测的定位信息
        self._update_position(t_locations_left, 0, self.state[0], rp_alarming)
        self._update_position(t_locations_right, 1, self.state[1], rp_alarming)

        # 左相机 定位点
        left_location = self._locations[0]
        # 右相机 定位点
        right_location = self._locations[1]

        left_zero = self._f_equal_zero(left_location)
        right_zero = self._f_equal_zero(right_location)

        # 左相机未检测到，右相机检测到的装甲板
        choose = np.logical_and(left_zero, np.logical_not(right_zero))
        try:
            left_location[choose] = right_location[choose]
        except Exception as e:
            print(choose)
        T = time.time()
        for i in range(1, 6):
            self._location[str(i)][0:3] = left_location[i - 1].tolist()
            # 当前未检测到的对应装甲板
            if self._location[str(i)][0:2] != [0, 0]:
                self._current_time[str(i)] = T

        if self._lp:
            self._location_prediction()

        # if self.reset_count == self.reset_thre:
        #     self.reset_count = 0
        #     for i in range(1, 6):  # 初始化位置为全零
        #         self._location[str(i)][0:3] = [0, 0, 0]
        #     # 前一帧位置为全零
        #     self._location_cache = self._location.copy()
        # else:
        #     self.reset_count += 1

    def _update_position(self, t_location, camera_type, detection_type, rp_alarming: dict):
        """
        根据一个相机的检测结果，在世界坐标系下更新装甲板位置
        :param t_location: the predicted locations [N,cls+x+y+depth]
        :param camera_type: 0: left 1: right
        :param detection_type: 0: radar 1: Delaunary 2: kd_tree
        """
        # 位置信息初始化，上次信息已保存至cache
        self._locations[camera_type].fill(0)

        locations = None

        if isinstance(t_location, np.ndarray):
            # 当t_location中有一数组对应z值为nan，对应false
            mask = np.logical_not(np.any(np.isnan(t_location), axis=1))
            locations = t_location[mask]

        if isinstance(locations, np.ndarray):
            pred_loc = []
            locations[1:3] = np.around(locations[1:3])
            for armor in range(1, 6):
                # 检测到装甲板
                if (locations[:, 0] == armor).any():
                    # 使用雷达点云定位
                    if not detection_type and not camera_type:
                        # 一次只找一个装甲板 拉成4维向量
                        l1 = locations[locations[:, 0]
                                       == armor].reshape(-1)
                        # 以下是对世界坐标的计算
                        K_C = np.linalg.inv(self._K_O)
                        C = (K_C @ np.concatenate([l1[1:3], np.ones(1)], axis=0).reshape(3, 1)) * l1[
                            3] * 1000   # 做了单位换算
                        B = np.concatenate(
                            [np.array(C).flatten(), np.ones(1)], axis=0)
                        l1[1:] = (self._T[0] @ B)[:3] / 1000
                        # 结合反投影模块信息判断车辆位置是否在指定区域内
                        for i in rp_alarming.keys():
                            if i == 'a_xxx_tou_a':
                                continue
                            # 反投影模块信息表明当前区域有对应编号车辆
                            if (rp_alarming[i] == armor).any():
                                # 与3维区域比较 结果相同则不使用德劳内
                                if is_inside_polygon(np.array(self._region[i])[:, :2], l1[1:3]):
                                    break
                                else:
                                    # 使用德劳内定位
                                    l1 = locations[locations[:, 0] == armor].reshape(-1)
                                    l1[1:] = self._loc_D[camera_type].get_point_pos(l1, detection_type).reshape(-1)
                            else:
                                continue

                    else:
                        l1 = locations[locations[:, 0]
                                       == armor].reshape(-1)
                        l1[1:] = self._loc_D[camera_type].get_point_pos(l1, detection_type).reshape(-1)

                    # 对应装甲板位置信息无效，跳过
                    if np.isnan(l1).any():
                        continue
                    # 异常值处理
                    if l1[1] > self.thre_predict[1] or l1[1] < self.thre_predict[0]:
                        continue
                    # 装甲板置信度处理
                    # 检测到，涨置信度
                    self._confidence[armor] = min(1.1, self._confidence[armor] + self.con_incre)
                    if self._confidence[armor] < self.con_thre:
                        continue
                    pred_loc.append(l1.reshape(-1))
                else:
                    # 装甲板未出现，置信度衰减处理
                    if self._confidence[armor] > self.con_thre:
                        self._confidence[armor] -= self.con_decre2
                    else:
                        self._confidence[armor] -= self.con_decre1
                        self._confidence[armor] = max(0., self._confidence[armor])
            # 更新定位信息
            if len(pred_loc):
                l_ = np.stack(pred_loc, axis=0)
                choose = l_[:, 0].reshape(-1).astype(np.int32)
                choose = choose - np.ones_like(choose)
                self._locations[camera_type][choose] = l_[:, 1:4].copy()

    def get_location(self):
        return np.array(list(i[0:2] for i in list(self._location.values())))[:5, :]

    def get_mode(self):
        return f"左相机定位：{self.state_name[self.state[0]]}\t"
        # f"右相机定位：{self.state_name[self.state[1]]}"

    def pc_location(self, camera_type, armor: np.ndarray):
        """
        显示点云定位
        """
        self._refresh()
        # 左相机，使用点云定位
        if not camera_type and not self.state[camera_type]:
            l1 = armor.reshape(-1)
            K_C = np.linalg.inv(self._K_O)
            C = (K_C @ np.concatenate([l1[1:3], np.ones(1)], axis=0).reshape(3, 1)) * l1[3] * 1000
            B = np.concatenate(
                [np.array(C).flatten(), np.ones(1)], axis=0)
            l1[1:] = (self._T[0] @ B)[:3] / 1000
            print(l1)
        else:
            # 德劳内（后两种定位均在其中）
            l1 = armor.reshape(-1)
            l1[1:] = self._loc_D[camera_type].get_point_pos(l1, self.state[camera_type]).reshape(-1)
        try:
            self._update({'1': l1[1:3]}, {'1': l1[1:3]})
            self._twinkle(self._region)
            self._show()
        except Exception as e:
            print(e)

    def get_draw(self, camera_type):
        """
        获取德劳内标点
        """
        return self._loc_D[camera_type].get_points()

    def get_last_loc(self) -> np.ndarray:
        """
        获取最新一次的位置信息
        :return: (5,4) ndarray
        """
        return np.array(list(self._last_location.values()))

    def show(self):
        """
        执行预警闪烁并画点显示地图
        """
        # _update函数内使用索引对位置的x，y值进行处理
        self._update(self._location, self._last_location)
        self._refresh()
        self._twinkle(self._region)
        self._show()
