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

    :param box:为凸四边形的四点 shape is (4,2)
    :param point:为需判断的是否在内的点 shape is (2,)
    """
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
        next_i = i + 1 if i + 1 < len(polygon) else 0
        x1, y1 = corner
        x2, y2 = polygon[next_i]
        if (x1 == px and y1 == py) or (x2 == px and y2 == py):  # 点在多边形顶点时
            is_in = True
            break
        if min(y1, y2) < py <= max(y1, y2):  # 过当前点的水平线将多边形该边两点分割
            x = x1 + (py - y1) * (x2 - x1) / (y2 - y1)
            if x == px:  # 点在多边形边上时
                is_in = True
                break
            elif x > px:  # 点在多边形当前线段的左侧
                is_in = not is_in

    return is_in


def armor_filter(armors: np.ndarray):
    """
    装甲板去重

    :param armors 格式定义： [N,bbox(xyxy),conf,cls,bbox(xyxy),conf,cls,col, N]

    :return: armors np.ndarray 每个id都最多有一个装甲板
    """
    # 直接取最高置信度
    ids = [1, 2, 3, 4, 5]
    if armors.shape[0] != 0:
        results = []
        for i in ids:
            mask = armors[:, 11] == i
            armors_mask = armors[mask]
            if armors_mask.shape[0]:
                armor = armors_mask[np.argmax(armors_mask[:, 9])]

                armor[6] = armor[0] + (armor[2] - armor[0]) * 0.4
                armor[7] = armor[1] + (armor[3] - armor[1]) * 0.75
                armor[8] = armor[0] + (armor[2] - armor[0]) * 0.6
                armor[9] = armor[1] + (armor[3] - armor[1]) * 0.85
                results.append(armor)
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
    mask_intensity = gray > intensity_thre
    frame_hsv = cv2.cvtColor(frame_m, cv2.COLOR_BGR2HSV)
    mask = np.logical_and(frame_hsv[:, :, 0] < h, frame_hsv[:, :, 0] > l)
    b, g, r = cv2.split(frame_m)
    # 通道差阈值过滤
    if red:
        mask_color = (r - b) > channel_thre
    else:
        mask_color = (b - r) > channel_thre
    frame_ii[np.logical_and(np.logical_and(mask, mask_color), mask_intensity)] = 255
    kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5))
    frame_ii = cv2.dilate(frame_ii, kernel)
    gray[frame_ii < 200] = 0
    contours, _ = cv2.findContours(gray, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    flag = False
    for c in contours:
        if cv2.contourArea(c) > 5:
            flag = True
    return flag
