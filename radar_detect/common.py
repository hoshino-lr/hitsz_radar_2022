"""
common.py
对于所有类都有用的函数
对上海交通大学源代码进行了一些删减
最终修改 by 李龙 2021/1/14
"""
import numpy as np
import cv2


def is_inside(box: np.ndarray, point: np.ndarray):
    """
    判断点是否在凸四边形中

    :param box:为凸四边形的四点 shape is (4,2)，按顺时针或逆时针次序输入
    :param point:为需判断的是否在内的点 shape is (2,)
    """
    assert box.shape == (4, 2)
    assert point.shape == (2,)
    # 计算凸四边形各点到待判断点的向量 以及凸四边形各点到相邻的下一个点的向量
    AM = point - box[0]
    AB = box[1] - box[0]
    BM = point - box[1]
    BC = box[2] - box[1]
    CM = point - box[2]
    CD = box[3] - box[2]
    DM = point - box[3]
    DA = box[0] - box[3]
    # 计算凸四边形各顶点到待判断点向量与到相邻下一点向量的叉积
    a = np.cross(AM, AB)
    b = np.cross(BM, BC)
    c = np.cross(CM, CD)
    d = np.cross(DM, DA)
    # 对各个顶点所求的叉积方向全相同，则点在凸四边形内
    return (a >= 0 and b >= 0 and c >= 0 and d >= 0) or \
           (a <= 0 and b <= 0 and c <= 0 and d <= 0)


def is_inside_polygon(polygon: np.ndarray, point: np.ndarray) -> bool:
    """
    判断点是否在多边形内，点的次序需要连续（顺、逆时针）

    :param polygon: [[], [], [], [], ...] 多边形的点 shape:(N,2)
    :param point: [x, y] 判断是否在内的点 shape:(2,)
    :return: is_in 点是否在多边形内
    """
    px, py = point
    is_in = False

    for i, corner in enumerate(polygon):
        # 循环每条边的两个端点
        next_i = i + 1 if i + 1 < len(polygon) else 0
        x1, y1 = corner
        x2, y2 = polygon[next_i]
        # 点在多边形顶点时，视为在多边形内
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):
            is_in = True
            break
        # 过当前点的水平线将多边形该边两点分割
        if min(y1, y2) < py <= max(y1, y2):
            # 做当前点在多边形该边上的水平投影
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # 点在多边形边上时
                is_in = True
                break
            elif x > px:  # 点在多边形当前线段的左侧
                is_in = not is_in   # 点在封闭多边形内，当且仅当其在奇数条线段的左侧

    return is_in


def res_decode(data: list) -> np.ndarray:
    """
    将识别小车的检测框列表转化成二维数组，并从存储宽高转变为存储右下角坐标
    :return: (N,14)的二维数组，第二维下标0~5是小车检测框的信息，6~13是装甲板检测框的信息
    """
    R = np.ones((len(data), 14)) * np.nan
    for i in range(len(data)):
        Car = data[i].car
        R[i, 0] = Car.x
        R[i, 1] = Car.y
        R[i, 2] = Car.width
        R[i, 3] = Car.height
        R[i, 4] = Car.confidence
        R[i, 5] = Car.type
        Armor = data[i].armor
        if Armor.type != 9:
            R[i, 6] = Armor.x
            R[i, 7] = Armor.y
            R[i, 8] = Armor.width
            R[i, 9] = Armor.height
            R[i, 10] = Armor.confidence
            R[i, 11] = Armor.type
            R[i, 12] = Armor.color
            R[i, 13] = i
    R[:, 2] = R[:, 2] + R[:, 0]
    R[:, 3] = R[:, 3] + R[:, 1]
    return R


def armor_filter(armors: np.ndarray):
    """
    装甲板去重

    :param armors 格式定义： [N,[bbox(xyxy),conf,cls,bbox(xyxy),conf,cls,col, N]]

    :return: armors np.ndarray 每个id都最多有一个装甲板
    """

    # 直接取最高置信度
    ids = [1, 2, 3, 4, 5]
    if armors.shape[0] != 0:
        results = []
        for i in ids:
            # 筛选id等于i的装甲板检测结果，置入armors_mask数组
            mask = armors[:, 11] == i
            armors_mask = armors[mask]
            if armors_mask.shape[0]:
                # 选取同id置信度最高的结果
                armor = armors_mask[np.argmax(armors_mask[:, 9])]

                # armor[6] = armor[0] + (armor[2] - armor[0]) * 0.4
                # armor[7] = armor[1] + (armor[3] - armor[1]) * 0.75
                # armor[8] = armor[0] + (armor[2] - armor[0]) * 0.6
                # armor[9] = armor[1] + (armor[3] - armor[1]) * 0.85
                results.append(armor)
        # 转化为ndarray类型二维数组存储，并返回
        if len(results):
            armors = np.stack(results, axis=0)
            for i in range(0, len(results)):
                armors[i][13] = i
        else:
            armors = np.array(results)
    return armors


def car_classify(frame_m, red=True):
    """
    亮度阈值加HSV判断车辆颜色

    :param frame_m:输入图像（可以是ROI)
    :param red:判断为红还是蓝

    :return: 判断结果
    """

    # 根据待检测的颜色初始化色调范围
    if red:
        l = 10
        h = 30
    else:
        l = 88
        h = 128
    intensity_thre = 200
    channel_thre = 150

    frame_ii = np.zeros((frame_m.shape[0], frame_m.shape[1]), dtype=np.uint8)
    # intensity threshold
    gray = cv2.cvtColor(frame_m, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    # 判断图像每个像素的颜色深度是否达标，过小则不计入
    mask_intensity = gray > intensity_thre
    frame_hsv = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
    # 判断HSV三通道图像中每个像素的色调H是否在开区间(l,h)内
    mask = np.logical_and(frame_hsv[:, :, 0] < h, frame_hsv[:, :, 0] > l)
    b, g, r = cv2.split(frame_m)
    # 通道差阈值过滤
    # 判断BGR三通道图像中每个像素的红色（或蓝色）特征是否明显
    if red:
        mask_color = (r - b) > channel_thre
    else:
        mask_color = (b - r) > channel_thre
    # 二值化，同时符合三次判断的像素点，在灰度矩阵frame_ii上绘制为255灰度
    frame_ii[np.logical_and(np.logical_and(mask, mask_color), mask_intensity)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    #
    frame_ii = cv2.dilate(frame_ii, kernel)
    # 非检测出来的红色（或蓝色）区域灰度设置为0，便于边缘检测
    gray[frame_ii < 200] = 0
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = False
    for c in contours:
        # 防止小块噪点被统计
        if cv2.contourArea(c) > 5:
            flag = True
    return flag