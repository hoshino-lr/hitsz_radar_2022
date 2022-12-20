'''
小地图绘制类
draw_map.py
整体部分将与上交开源代码相似或相同
'''
import cv2
import numpy as np

from config import MAP_PATH, map_size, enemy2color


class CompeteMap(object):
    """
    小地图绘制类
    draw_map.py

    使用顺序 twinkle->update->show->refresh
    """
    # 类内变量
    _circle_size = 10
    _twinkle_times = 29  # 闪烁次数

    def __init__(self, region, real_size, enemy, api):
        """
        :param region:预警区域
        :param real_size:真实赛场大小
        :param enemy:敌方编号
        :param api:显示api f(img)
        """
        self._enemy = enemy
        # map为原始地图(canvas),out_map在每次refresh时复制一份canvas副本并在其上绘制车辆位置及预警
        self._map = cv2.imread(MAP_PATH)
        self._map = cv2.resize(self._map, map_size)
        # 显示api调用（不要跨线程）
        self._show_api = api
        self._real_size = real_size  # 赛场实际大小
        self._draw_region(region)
        # 闪烁画面，out_map前一步骤，因为out_map翻转过，而region里面（x,y）依照未翻转的坐标定义，若要根据region进行闪烁绘制，用未翻转地图更加方便
        self._out_map_twinkle = self._map.copy()
        # 画点以后画面
        if self._enemy:
            # enemy is blue,逆时针旋转90
            self._out_map = cv2.rotate(
                self._out_map_twinkle, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            # enemy is red,顺时针旋转90
            self._out_map = cv2.rotate(
                self._out_map_twinkle, cv2.ROTATE_90_CLOCKWISE)

        self._twinkle_event = {}

    def _refresh(self):
        """
        闪烁画面复制canvas,刷新
        """
        self._out_map_twinkle = self._map.copy()

    def _update(self, location: dict, last_location: dict):
        """
        更新车辆位置
        :param last_location:车辆位置字典 索引为'1'-'5',内容为车辆位置数组(2,)
        :param location:车辆位置字典 索引为'1'-'5',内容为车辆位置数组(2,)
        """
        if self._enemy:
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_COUNTERCLOCKWISE)
        else:
            self._out_map = cv2.rotate(self._out_map_twinkle, cv2.ROTATE_90_CLOCKWISE)
        _loc_map = [0, 0]
        for armor in location.keys():
            mode = 1
            ori_x = int(location[armor][0] / self._real_size[0] * map_size[0])
            ori_y = int((self._real_size[1] - location[armor][1]) / self._real_size[1] * map_size[1])
            if not ori_x:
                mode = 0
                ori_x = int(last_location[armor][0] / self._real_size[0] * map_size[0])
                ori_y = int((self._real_size[1] - last_location[armor][1]) / self._real_size[1] * map_size[1])
            # 位置翻转
            if self._enemy:
                _loc_map = ori_y, map_size[0] - ori_x
            else:
                _loc_map = map_size[1] - ori_y, ori_x
            # 画定位点；armor为字符'1'-'5'
            self._draw_circle(_loc_map, int(armor), dtype=mode)

    def _show(self):
        """
        调用show_api展示
        """
        self._show_api(self._out_map)

    def _draw_region(self, region: dict):
        """
        param region
        在canvas绘制预警区域（canvas 原始地图 _map）
        """
        for r in region.keys():
            alarm_type, team, _, _ = r.split('_')
            if (alarm_type == 'm' or alarm_type == 'a') and team == enemy2color[self._enemy]:  # 预警类型判断，若为map或all类型
                # n点预警
                f = lambda x: (int(x[0] * map_size[0] // self._real_size[0]),
                               int((self._real_size[1] - x[1]) * map_size[1] // self._real_size[1]))
                rect = np.array(region[r]).reshape(-1, 3)[:, :2]
                count = len(region[r])
                if count > 2:
                    for i in range(count):
                        cv2.line(self._map, f(rect[i]), f(rect[(i + 1) % count]), (0, 255, 0), 2)

    def _draw_circle(self, location, armor: int, dtype: int):
        """
        画定位点
        """
        img = self._out_map
        color = (255 * self._enemy, 0, 255 * (1 - self._enemy))  # 解算颜色
        last_color = (127 * self._enemy, 0, 127 * (1 - self._enemy))  # 解算颜色
        armor = armor - 5 * (armor > 5)  # 将编号统一到1-5
        if dtype:
            cv2.circle(img, tuple(location), self._circle_size, color, -1)  # 内部填充
        else:
            cv2.circle(img, tuple(location), self._circle_size, last_color, -1)  # 内部填充
        cv2.circle(img, tuple(location), self._circle_size, (0, 0, 0), 1)  # 外边框
        # 数字
        cv2.putText(img, str(armor),
                    (location[0] - 7 * self._circle_size // 10, location[1] + 6 * self._circle_size // 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.6 * self._circle_size / 10, (255, 255, 255), 2)

    def _add_twinkle(self, region: str):
        """
        预警事件添加函数
        :param region:要预警的区域名
        """
        if region not in self._twinkle_event.keys():
            # 若不在预警字典内，则该事件从未被预警过，添加预警事件
            # _twinkle_times 类内变量，初始为29
            self._twinkle_event[region] = self._twinkle_times
        else:
            self._twinkle_event[region] = self._twinkle_times - 9 + (self._twinkle_event[region] % 10)

    def _twinkle(self, region):
        """
        闪烁执行类

        :param region:所有预警区域
        """
        # 对_twinkle_event字典进行遍历，r为字典中的键
        for r in self._twinkle_event.keys():
            if self._twinkle_event[r] == 0:
                # 不能再预警
                continue
            if self._twinkle_event[r] % 10 >= 5:
                x = np.float32(region[r]).reshape(-1, 3)[:, :2]
                x[:, 0] = (x[:, 0] * map_size[0] // self._real_size[0])
                x[:, 1] = ((self._real_size[1] - x[:, 1]) * map_size[1] // self._real_size[1])
                rect = x.astype(int)
                cv2.fillConvexPoly(self._out_map_twinkle, rect, (0, 0, 255))

            # 减少预警次数
            self._twinkle_event[r] -= 1
