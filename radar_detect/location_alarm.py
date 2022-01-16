'''
位置预警类，还需要修改和完善
created by 黄继凡 2021/12
最新修改 by 黄继凡 2021/1/15 
'''
import cv2
import numpy as np
from numpy.lib.arraysetops import isin

import mapping.draw_map as draw_map #引入draw_map模块，使用其中的CompeteMap类
from resources.config import armor_list, color2enemy, enemy_case,cam_config

# 此函数来自上交源码的common.py文件，可考虑删除，通过模块引入使用该函数
def is_inside(box: np.ndarray, point: np.ndarray):
    '''
    判断点是否在凸四边形中

    :param box:为凸四边形的四点 shape is (4,2)
    :param point:为需判断的是否在内的点 shape is (2,)
    '''
    assert box.shape == (4, 2)
    assert point.shape == (2,)
    AM = point - box[0]
    AB = box[1] - box[0]
    BM = point - box[1]
    BC = box[2] - box[1]
    CM = point - box[2]
    CD = box[3] - box[2]
    DM = point - box[3]
    DA = box[0] - box[3]
    a = np.cross(AM, AB)
    b = np.cross(BM, BC)
    c = np.cross(CM, CD)
    d = np.cross(DM, DA)
    return a >= 0 and b >= 0 and c >= 0 and d >= 0 or \
           a <= 0 and b <= 0 and c <= 0 and d <= 0

class Alarm(draw_map.CompeteMap):
    '''
    预警类，继承自地图画图类
    v2:
    删除原先的refresh、show函数，
    two_camera_merge_update、update函数更改传入参数、
    返回车辆位置字典_location，统一绘图
    '''
    # param

    _pred_time = 10  # 预测几次
    _pred_ratio = 0.2  # 预测速度比例

    _ids = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5}  # 装甲板编号到标准编号
    _lp = True  # 是否位置预测
    _z_a = True  # 是否进行z轴突变调整
    _z_thre = 0.2  # z轴突变调整阈值
    _ground_thre = 100  # 地面阈值，我们最后调到了100就是没用这个阈值，看情况调
    _using_l1 = True  # 不用均值，若两个都有预测只用右相机预测值

    def __init__(self, region: dict, api, touch_api, enemy, real_size, two_Camera=True, debug=False):
        '''

        :param region:预警区域
        :param api:主程序显示api，传入画图程序进行调用（不跨线程使用,特别是Qt）
        :param touch_api:车间通信api
        :param enemy:敌方编号
        :param real_size:场地实际大小
        :param two_camera:是否使用两个相机
        :param debug:debug模式
        '''
        super(Alarm, self).__init__(region, real_size, enemy, api)

        self._region = region
        # location 车辆位置字典，键为字符'1'-'10'，值为车辆的位置数组
        self._location = {}
        if two_Camera:
            # 分别为z坐标缓存，相机世界坐标系位置，以及（相机到世界）转移矩阵
            self._z_cache = [None, None]
            self._camera_position = [None, None]
            self._T = [None, None]
        else:
            self._z_cache = [None]
            self._camera_position = [None]
            self._T = [None]
        self._K_O = cam_config['cam_left']['K_0']
        self._location_pred_time = np.zeros(10, dtype=int)  # 预测次数记录
        self._enemy = enemy
        self._touch_api = touch_api
        self._debug = debug
        self._two_camera = two_Camera

        # 判断x各行是否为全零的函数
        self._f_equal_zero = lambda x: np.isclose(np.sum(x, axis=1), np.zeros(x.shape[0]))
        for i in range(1, 11): # 初始化位置为全零
            self._location[str(i)] = [0, 0]
         # 前两帧位置为全零
        self._location_cache = [self._location.copy(), self._location.copy()]

    def push_T(self, T, camera_position, camera_type):
        '''
        位姿信息

        :param T:相机到世界转移矩阵
        :param camera_position:相机在世界坐标系坐标
        :param camera_type:相机编号，若为单相机填0
        '''
        if camera_type > 0 and not self._two_camera:
            return

        self._camera_position[camera_type] = camera_position.copy()
        self._T[camera_type] = T.copy()

    def _check_alarm(self):
        '''
        预警检测

        alarming:各区域是否有预警;
        base_alarming:基地是否有预警
        '''
        alarming = False 
        base_alarming = False
        for loc in self._region.keys():
            alarm_type, shape_type, team, target, l_type = loc.split('_')
            targets = []
            for armor in list(self._location.keys())[0 + self._enemy * 5:5+ self._enemy * 5]:  # 检测敌方
                l = np.float32(self._location[armor])
                if alarm_type == 'm' or alarm_type == 'a':  # 若为位置预警
                    if shape_type == 'r' and (target not in enemy_case or color2enemy[team] == self._enemy):  # 对于特殊地点，只考虑对敌方进行预警
                        # 矩形区域采用范围判断
                        if l[0] >= self._region[loc][0] and l[1] >= self._region[loc][3] and \
                                l[0] <= self._region[loc][2] and l[1] <= self._region[loc][1]:
                            targets.append(int(armor) -1 ) 
                    if shape_type == 'l' and color2enemy[team] != self._enemy:  # base alarm
                        # 直线检测
                        up_p = np.float32(self._region[loc][:2])  # 上端点
                        dw_p = np.float32(self._region[loc][2:4])  # 下端点
                        dis_thres = self._region[loc][4]
                        up_l = up_p - dw_p  # 直线向上的向量
                        dw_l = dw_p - up_p  # 直线向下的向量
                        m_r = np.float32([up_l[1], -up_l[0]])  # 方向向量，向右
                        m_l = np.float32([-up_l[1], up_l[0]])  # 方向向量，向左
                        f_dis = lambda m: m @ (l - dw_p) / np.linalg.norm(m)  # 计算从下端点到物体点在各方向向量上的投影
                        if l_type == 'l':
                            dis = f_dis(m_l)
                        if l_type == 'r':
                            dis = f_dis(m_r)
                        if l_type == 'a':
                            dis = abs(f_dis(m_r))  # 绝对距离
                        # 当物体位置在线段内侧，且距离小于阈值时，预警
                        if up_l @ (l - dw_p) > 0 and dw_l @ (l - up_p) > 0 and\
                                dis_thres >= dis >= 0:
                            targets.append(int(armor) - 1)
                    if shape_type == 'fp' and (target not in enemy_case or color2enemy[team] == self._enemy):
                        # 判断是否在凸四边形内
                        if is_inside(np.float32(self._region[loc][:8]).reshape(4, 2),point = l):
                            targets.append(int(armor) - 1)

            if len(targets):
                # 发送预警
                if alarm_type == 'l':
                    # 基地预警发送，编码规则详见主程序类send_judge
                    base_alarming = True
                    self._touch_api({'task': 3, 'data': [targets]})
                else:
                    super(Alarm, self)._add_twinkle(loc)
                    alarming = True
                    if target in ['feipo', 'feipopre', 'gaodipre', 'gaodipre2']:
                        self._touch_api({'task': 2, 'data': [team, target, targets]})

        return alarming, base_alarming

    def _adjust_z_one_armor(self, l, camera_type):
        '''
        z轴突变调整，仅针对一个装甲板

        :param l:(cls+x+y+z) 一个id的位置
        :param camera_type:相机编号
        '''
        if isinstance(self._z_cache[camera_type], np.ndarray):
            mask = np.array(self._z_cache[camera_type][:, 0] == l[0])  # 检查上一帧缓存z坐标中有没有对应id
            if mask.any():
                z_0 = self._z_cache[camera_type][mask][:, 1]
                if z_0 < self._ground_thre:  # only former is on ground do adjust
                    z = l[3]
                    if z - z_0 > self._z_thre:  # only adjust the step from down to up
                        # 以下计算过程详见技术报告公式
                        ori = l[1:].copy()
                        line = l[1:] - self._camera_position[camera_type]
                        ratio = (z_0 - self._camera_position[camera_type][2]) / line[2]
                        new_line = ratio * line
                        l[1:] = new_line + self._camera_position[camera_type]
                        if self._debug:
                            # z轴变换debug输出
                            # print('{0} from'.format(armor_list[(self._ids[int(l[0])]) - 1]), ori, 'to', l[1:])
                            print('{0} from'.format(armor_list[int(l[0]) - 1]), ori, 'to', l[1:])


    def show(self):
        '''
        执行预警闪烁并画点显示地图
        '''
        super(Alarm, self)._twinkle(self._region)
        super(Alarm, self)._update(self._location)
        super(Alarm, self)._show()

    def _location_prediction(self):
        '''
        位置预测
        '''
        
        # 次数统计
        # 若次数为1则不能预测，此时time_equal_one对应元素为1 time_equal_zero对应元素为0
        time_equal_one = self._location_pred_time == 1
        time_equal_zero = self._location_pred_time == 0

        # 上两帧位置 (2,N)
        pre = np.stack([np.float32(list(self._location_cache[0].values())),
                        np.float32(list(self._location_cache[1].values()))], axis=0)
        # 该帧预测位置
        now = np.float32(list(self._location.values()))

        pre2_zero = self._f_equal_zero(pre[0])  # the third latest frame 倒数第二帧
        pre1_zero = self._f_equal_zero(pre[1])  # the second latest frame 倒数第一帧
        now_zero = self._f_equal_zero(now)  # the latest frame 当前帧

        # 仅对该帧全零，上两帧均不为0的id做预测
        do_prediction = np.logical_and(
            np.logical_and(
                np.logical_and(np.logical_not(pre2_zero), np.logical_not(pre1_zero)),now_zero), 
            np.logical_not(time_equal_one))
        v = self._pred_ratio * (pre[1] - pre[0])  # move vector between frame

        if self._debug:
            # 被预测id,debug输出
            for i in range(10):
                if do_prediction[i]:
                    print("{0} lp yes".format(armor_list[i]))

        now[do_prediction] = v[do_prediction] + pre[1][do_prediction]

        set_time = np.logical_and(do_prediction, time_equal_zero)  # 次数为0且该帧做预测的设置为最大次数
        reset = np.logical_and(np.logical_not(now_zero), time_equal_one)  # 对当前帧不为0，且次数为1的进行次数重置
        self._location_pred_time[reset] = 0
        self._location_pred_time[set_time] = self._pred_time + 1
        self._location_pred_time[do_prediction] -= 1  # 对做预测的进行次数衰减

        # 预测填入
        for i in range(1, 11):
            self._location[str(i)] = now[i - 1].tolist()

        # push new data
        self._location_cache[0] = self._location_cache[1].copy()
        self._location_cache[1] = self._location.copy()

    def check(self):
        '''
        预警检测
        '''
        alarming, base_alarming = self._check_alarm()

        return alarming, base_alarming

    def two_camera_merge_update(self, locations):
        """
        两个相机合并更新，顾名思义，two_camera为True才能用的专属api

        :param locations: the list of the predicted locations [N,cls+x+y+z] of the both two cameras
        :param radar:  the list of the radar class corresponding to the two camera
        """
        if self._two_camera:
            # init location
            for i in range(1, 11):
                self._location[str(i)] = [0, 0]
            rls = []
            for location in locations:# 车辆预测框列表，列表元素格式为[N,cls+x+y+z]
                #对左右两相机获取结果进行处理
                if isinstance(location, np.ndarray):
                    # 滤除nan
                    mask = np.logical_not(np.any(np.isnan(location), axis=1)) # 当location中有一数组对应z值为nan，对应false
                    # rls对应格式为(N,cls+x0+y0+z)，只添加location中z值存在的对应数据
                    rls.append(location[mask])
                else:
                    rls.append(None)

            
            pred_loc = [] # 存储预测的位置 cls+x+y+z
            if self._z_a:
                pred_1 = []
                pred_2 = []
            
            # for armor in self._ids.keys():
            for armor in range(1,11):
                l1 = None  # 对于特定id，第一个相机基于直接神经网络预测装甲板计算出的位置
                l2 = None  # 对于特定id，第二个相机基于直接神经网络预测装甲板计算出的位置
                al1 = None  # 对于特定id，第一个相机预测出的位置
                al2 = None  # 对于特定id，第二个相机预测出的位置

                # 第一个相机数据处理
                if isinstance(rls[0], np.ndarray):
                    mask = rls[0][:, 0] == armor
                    if mask.any(): # 预测装甲板编号有对应
                        l1 = rls[0][mask].reshape(-1)
                        # 坐标换算为世界坐标
                        l1[1:] = (self._T[0] @ np.concatenate([np.concatenate([l1[1:3], np.ones(1)], axis=0) * 
                                    l1[3], np.ones(1)], axis=0))[:3]
                        # z坐标解算
                        if self._z_a:
                            self._adjust_z_one_armor(l1, 0)
                        al1 = l1
                # 第二个相机处理
                if isinstance(rls[1], np.ndarray):
                    mask = rls[1][:, 0] == armor
                    if mask.any():
                        l2 = rls[1][mask].reshape(-1)
                        l2[1:] = (self._T[1] @ np.concatenate([np.concatenate([l2[1:3], np.ones(1)], axis=0)*
                                    l2[3], np.ones(1)], axis=0))[:3]
                        if self._z_a:
                            self._adjust_z_one_armor(l2, 1)
                            al2 = l2

                if self._z_a:
                    if isinstance(al1, np.ndarray):
                        pred_1.append(al1[[0, 3]]) # cache cls+z
                    if isinstance(al2, np.ndarray):
                        pred_2.append(al2[[0, 3]])

                # 数据融合
                armor_pred_loc = None
                if isinstance(l1, np.ndarray):
                    armor_pred_loc = l1.reshape(-1)
                if isinstance(l2, np.ndarray):
                    if isinstance(armor_pred_loc, np.ndarray): # 只有左相机有数据，以左相机为准
                        if not self._using_l1:
                            # 若_using_l1为真，则不取均值，以右相机为准
                            armor_pred_loc = (armor_pred_loc + l2.reshape(-1)) / 2 
                    else:
                        armor_pred_loc = l2.reshape(-1)

                if isinstance(armor_pred_loc, np.ndarray):
                    pred_loc.append(armor_pred_loc)
                    
            # z cache
            if self._z_a:
                if len(pred_1):
                    self._z_cache[0] = np.stack(pred_1, axis=0)
                else:
                    self._z_cache[0] = None
                if len(pred_2):
                    self._z_cache[1] = np.stack(pred_2, axis=0)
                else:
                    self._z_cache[1] = None
            
            # 发送裁判系统小地图
            judge_loc = {}
            if len(pred_loc):
                pred_loc = np.stack(pred_loc, axis=0)
                pred_loc[:, 2] = self._real_size[1] + pred_loc[:, 2] # 坐标变换，平移
                for i, armor in enumerate(pred_loc[:, 0]):
                    self._location[str(armor)] = pred_loc[i, 1:3].tolist() # 类成员只存(x,y)信息
                    judge_loc[str(armor)] = pred_loc[i, 1:].tolist() # 发送包存三维信息
            location = {}
            # 位置预测
            if self._lp:
                self._location_prediction()
            if self._debug:
                # 位置debug输出
                for armor, loc in judge_loc.items():
                    print("{0} in ({1:.3f},{2:.3f},{3:.3f})".format(armor_list[int(armor) - 1], *loc))
            for i in range(1, 11):
                location[str(i)] = self._location[str(i)].copy()
            
            # 执行裁判系统发送
            # judge_loc为未预测的位置，作为logging保存，location为预测过的位置，作为小地图发送
            self._touch_api({'task':1, 'data':[judge_loc, location]})

            # 返回车辆位置字典
            return self._location
            
        else:
            print('[ERROR] This update function only supports two_camera case, using update instead.')

    def update(self, t_location):
        '''
        
        #单相机使用
        :param t_location: the predicted locations [N,cls+x+y+z]
        :param radar:the radar class

        '''

        if not self._two_camera:

            # 位置信息初始化，上次信息已保存至cache
            for i in range(1, 11):
                self._location[str(i)] = [0, 0]
            
            locations = None
            if isinstance(t_location, np.ndarray):
                mask = np.logical_not(np.any(np.isnan(t_location), axis = 1)) #  当t_location中有一数组对应z值为nan，对应false
                locations = t_location[mask]

            judge_loc = {}
            if isinstance(locations, np.ndarray):
                pred_loc = []
                if self._z_a:
                    cache_pred = []
                # for armor in self._ids.keys():
                for armor in range(1, 11):
                    if(locations[:, 0] == armor).any():
                        l1 = locations[locations[:, 0] == armor].reshape(-1)
                        K_C = np.linalg.inv(self._K_O)
                        C = (K_C @ np.concatenate([l1[1:3], np.ones(1)], axis=0).reshape(3,1))* l1[3]*1000
                        B = np.concatenate([np.array(C).flatten() , np.ones(1)], axis=0)
                        l1[1:] = (self._T[0] @ B)[:3]/1000
                        l1[1] += 3.379
                        
                        if self._z_a:
                            self._adjust_z_one_armor(l1, 0)
                            cache_pred.append(l1[[0, 3]])
                        pred_loc.append(l1.reshape(-1))
                if len(pred_loc):
                    l = np.stack(pred_loc, axis=0)
                    cls = l[:, 0].reshape(-1, 1)
                    # z cache
                    if self._z_a:
                        self._z_cache[0] = np.stack(cache_pred, axis=0)

                    for i,armor in enumerate(cls):
                        self._location[str(int(armor))] = l[i, 1:3].tolist()
                        judge_loc[str(armor)] = l[i, 1:].tolist()

            if self._lp:
                self._location_prediction()
                # 执行裁判系统发送
            location = {}

            if self._debug:
                # 位置debug输出
                # for armor, loc in judge_loc.items():
                #     print("{0} in ({1:.3f},{2:.3f},{3:.3f})".format(armor_list[int(armor) - 1], *loc))
                pass
            for i in range(1, 11):
                location[str(i)] = self._location[str(i)].copy()
            # self._touch_api({'task': 1, 'data': [judge_loc, location]})
            # 返回车辆位置字典
            return self._location

        else:
            print(
                '[ERROR] This update function only supports single_camera case, using two_camera_merge_update instead.')


if __name__=="__main__":
    from radar_detect.Linar import DepthQueue
    from resources.config import cam_config,test_region,enemy_color,\
        real_size
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2
    import rosbag
    alarm = Alarm(test_region,[],[],enemy_color,real_size,False,True)
    cv2.namedWindow("img", cv2.WINDOW_NORMAL)  # 显示实际图片
    bag_file = '/home/hoshino/CLionProjects/camera_lidar_calibration/data/game/beijing.bag'
    bag = rosbag.Bag(bag_file, "r")
    topic = '/livox/lidar'
    bag_data = bag.read_messages(topic)
    K_0 = cam_config['cam_left']['K_0']
    C_0 = cam_config['cam_left']['C_0']
    E_0 = cam_config['cam_left']['E_0']
    rvec = cam_config['cam_left']['rvec']
    tvec = cam_config['cam_left']['tvec']
    depth = DepthQueue(100, size=[1024, 1024], K_0=K_0, C_0=C_0, E_0=E_0)
    T = np.eye(4)
    T[:3, :3] = cv2.Rodrigues(rvec)[0]  # 旋转向量转化为旋转矩阵
    T[:3, 3] = tvec.reshape(-1)  # 加上平移向量
    T = np.linalg.inv(T)  # 矩阵求逆
    alarm.push_T(T, (T @ (np.array([0, 0, 0, 1])))[:3],0)
    cloud = [[], [], [], []]
    for topic, msg, t in bag_data:
        pc = np.float32(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)).reshape(
            -1, 3)
        dist = np.linalg.norm(pc, axis=1)
        pc = pc[dist > 0.4]  # 雷达近距离滤除
        depth.push_back(pc)
    ori = cv2.imread("/home/hoshino/CLionProjects/hitsz_radar/resources/beijing.png")
    frame = ori.copy()
    rect = cv2.selectROI("img", frame, False)
    key = cv2.waitKey(1)
    while key != ord('q'):
        key = cv2.waitKey(1)
        # 分别在实际相机图和深度图上画ROI框来对照
        cv2.rectangle(frame, (rect[0], rect[1]), (rect[0] + rect[2], rect[1] + rect[3]), (0, 255, 0), 3)
        cv2.imshow("img", frame)
        if key == ord('r') & 0xFF:
            # 重选区域
            rect = cv2.selectROI("img", frame, False)
            frame = ori.copy()
        if key == ord('s') & 0xFF:
            # 显示世界坐标系和相机坐标系坐标和深度，以对测距效果进行粗略测试
            dee = depth.detect_depth([rect])
            location = np.zeros((10,4)).astype(np.int32)*np.nan
            location[6] = np.array([6,rect[0],rect[1],dee[0]]).astype(np.int32)
            points = alarm.update(location)
            print(points)
            
            

            