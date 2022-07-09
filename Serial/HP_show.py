'''
通过串口绘制血量类
'''
import numpy as np
import cv2

from resources.config import enemy2color, unit_list


class HP_scene(object):
    _stage_max = 5  # 5格血量条
    _size = (400, 310)
    _font_size = 0.8
    _outpost = 1500  # 前哨站血量上限
    _base = 5000  # 基地血量上限
    _guard = 600  # 哨兵血量上限
    _font = cv2.FONT_HERSHEY_TRIPLEX

    def __init__(self, enemy, show_api):
        """
        展示一个血量条

        :param enemy:enemy number
        :param show_api:在主UI显示调用的api f(img:np.ndarray)
        """
        self._scene = np.ones((self._size[1], self._size[0], 3), dtype=np.uint8)
        self._scene *= 240
        self._show_api = show_api
        self._enemy = enemy
        self._puttext = lambda txt, x, y, color: cv2.putText(self._scene, txt, (x, y), self._font,
                                                             self._font_size, color, 1, cv2.LINE_AA)
        # init title
        if enemy2color[self._enemy] == 'red':
            self.our_color = (255, 0, 0)
            self.enemy_color = (0, 0, 255)
        else:
            self.our_color = (0, 0, 255)
            self.enemy_color = (255, 0, 0)
        self._puttext("OUR {0}".format(enemy2color[not self._enemy]), 5, 28, self.our_color)
        self._puttext("ENEMY {0}".format(enemy2color[self._enemy]), 210, 28, self.enemy_color)
        # OUR CAR
        for i in range(8):
            self._puttext("{0}".format(unit_list[i + 8 * (not self._enemy)]), 5, 60 + 30 * i, self.our_color)
        # enemy
        for i in range(8):
            self._puttext("{0}".format(unit_list[i + 8 * (self._enemy)]), 210, 60 + 30 * i, self.enemy_color)
        # 划分线
        cv2.line(self._scene, (0, 32), (self._size[0], 32), (169, 169, 169), 3)
        cv2.line(self._scene, (200, 0), (200, 275), (169, 169, 169), 3)
        cv2.line(self._scene, (0, 185), (self._size[0], 185), (169, 169, 169), 3)
        cv2.line(self._scene, (0, 275), (self._size[0], 275), (169, 169, 169), 3)
        self._out_scene = self._scene.copy()

    def _put_hp(self, hp, hp_max, x, y):
        # 血量条长度MAX 100pixel
        hp_m = int(hp_max)
        if not hp_m:
            hp_m = 100
        radio = hp / hp_m
        if radio > 0.6:
            # 60%以上绿色
            color = (0, 255, 0)
        elif radio > 0.2:
            # 20%以上黄色
            color = (0, 165, 255)
        else:
            # 20%以下红色
            color = (128, 0, 128)
        # 画血量条
        width = 100 // self._stage_max
        cv2.rectangle(self._out_scene, (x, y), (x + int(100 * radio), y + 15), color, -1)
        # 画血量格子
        # for i in range(self._stage_max):
        cv2.rectangle(self._out_scene, (x, y), (x + 5 * width, y + 15), (0, 0, 0), 1)

    def update(self, HP, max_hp):
        """
        根据读取到的血量和计算的血量上限，绘制血量信息
        """
        for i in range(8):
            if i < 5:  # 我方
                hp = HP[i + 8 * (not self._enemy)]
                self._put_hp(hp, max_hp[i + 5 * (not self._enemy)], 60, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.our_color, 1, cv2.LINE_AA)
            if i == 5:  # guard
                hp = HP[i + 8 * (self._enemy)]
                self._put_hp(hp, self._guard, 60, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.our_color, 1, cv2.LINE_AA)
            if i == 6:
                hp = HP[i + 8 * (not self._enemy)]
                self._put_hp(hp, self._outpost, 60, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.our_color, 1, cv2.LINE_AA)
            if i == 7:
                hp = HP[i + 8 * (not self._enemy)]
                self._put_hp(hp, self._base, 60, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.our_color, 1, cv2.LINE_AA)

        # enemy
        for i in range(8):
            if i < 5:  # 1-5
                hp = HP[i + 8 * (self._enemy)]
                self._put_hp(hp, max_hp[i + 5 * self._enemy], 60 + 200, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165 + 200, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.enemy_color, 1, cv2.LINE_AA)
            if i == 5:  # guard
                hp = HP[i + 8 * (self._enemy)]
                self._put_hp(hp, self._guard, 60 + 200, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165 + 200, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.enemy_color, 1, cv2.LINE_AA)
            if i == 6:
                hp = HP[i + 8 * (self._enemy)]
                self._put_hp(hp, self._outpost, 60 + 200, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165 + 200, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.enemy_color, 1, cv2.LINE_AA)
            if i == 7:
                hp = HP[i + 8 * (self._enemy)]
                self._put_hp(hp, self._base, 60 + 200, 42 + 30 * i)
                cv2.putText(self._out_scene, "{0}".format(hp), (165 + 200, 56 + 30 * i), cv2.FONT_ITALIC,
                            0.5, self.enemy_color, 1, cv2.LINE_AA)

    def update_stage(self, stage, remain_time, BO, BO_max):
        """
        显示比赛阶段和BO数
        """
        cv2.putText(self._out_scene, "{0} {1}s      BO:{2}/{3}".format(stage, remain_time, BO, BO_max), (20, 300),
                    self._font,
                    0.6, (0, 0, 0), 2)

    def show(self):
        """
        和其他绘制类一样，显示
        """
        self._show_api(self._out_scene)

    def refresh(self):
        """
        和其他绘制类一样，换原始未绘制的画布，刷新
        """
        self._out_scene = self._scene.copy()


if __name__ == "__main__":
    hp_scene = HP_scene(0, [])
    hp = np.array([300, 400, 100, 100, 100, 100, 100, 100, 100, 400, 100, 100, 100, 100, 100, 100])
    max = np.array([500, 500, 500, 500, 500, 500, 500, 500, 500, 500])
    hp_scene.update(hp, max)
    hp_scene.update_stage('start', 10, 1, 5)
    cv2.namedWindow("HP", cv2.WINDOW_NORMAL)
    pic = hp_scene.show()
    cv2.imshow("HP", pic)
    cv2.waitKey()
