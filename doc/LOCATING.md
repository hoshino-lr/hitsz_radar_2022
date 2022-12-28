# 雷达站 2022 定位方案

**雷达站的主要功能是在较大范围内定位敌方车辆的位置，并完成预警等功能。我们只需要车在哪个区域的信息，所以定位的精度或许并不是特别关键。相反，定位的实时性和错误率可能会更为重要。**

---

## 目前方案（主要在 `radar_detect` 中)

### 共用部分

第一层神经网络得出装甲板位置，返回矩形框。
第二层神经网络得出装甲板序号。

然后用 `radar_detect/common.py` 中定义的 `armor_filter` 进行去重。

```python
def armor_filter(armors: np.ndarray):
    """
    装甲板去重

    :param armors 格式定义： [N,[bbox(xyxy),conf,cls,bbox(xyxy),conf,cls,col, N]]

    :return: armors np.ndarray 每个 id 都最多有一个装甲板
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

                # armor[6] = armor[0] + (armor[2] - armor[0]) * 0.4
                # armor[7] = armor[1] + (armor[3] - armor[1]) * 0.75
                # armor[8] = armor[0] + (armor[2] - armor[0]) * 0.6
                # armor[9] = armor[1] + (armor[3] - armor[1]) * 0.85
                results.append(armor)
        if len(results):
            armors = np.stack(results, axis=0)
            for i in range(0, len(results)):
                armors[i][13] = i
        else:
            armors = np.array(results)
    return armors
```

最后用同文件中定义的函数 `car_classify` 进行颜色判定。

```python
def car_classify(frame_m, red=True):
    """
    亮度阈值加 HSV 判断车辆颜色

    :param frame_m:输入图像（可以是 ROI)
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
```

---

### 主要方案：激光雷达测距

加载雷达点云，并对点云进行简单处理，如过近的点进行过滤。(`radar_detect/Linar.py` 中定义的 `Radar` 类)

```python
def preload(self):
    """
    预加载雷达点云，debug 用
    """
    lidar_bag = rosbag.Bag(BAG_FIRE, "r")
    topic = '/livox/lidar'
    bag_data = lidar_bag.read_messages(topic)
    for topic, msg, t in bag_data:
        pc = np.float32(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)).reshape(
            -1, 3)
        dist = np.linalg.norm(pc, axis=1)
        pc = pc[dist > 0.4]  # 雷达近距离滤除
        self.__queue[self._no].push_back(pc)
```

取点云在相机坐标系中的 z 坐标和在像素坐标系中的坐标，用队列进行储存 (`radar_detect/Linar.py` 中定义的 `DepthQueue` 类)

**考虑到单幅点云十分稀疏，深度图由队列中的多幅点云积分得到。每幅点云入队后即对深度图进行更新，对队列中所有点云 z 值取 `np.nanmedian`，即忽略 NaN 对像素坐标重合的点取中位数，赋值给深度图对应像素坐标的深度。点云出队后，将深度图中对应像素坐标的点的深度值置为 NaN。**

```python
def push_back(self, entry):

    # 当队列为空时，说明该类正在被初始化，置位初始化置位符
    if self.queue.empty():
        self.init_flag = True

    # 坐标转换 由雷达坐标转化相机坐标，得到点云各点在相机坐标系中的 z 坐标
    dpt = (self.E_0 @ (np.concatenate([entry, np.ones((entry.shape[0], 1))], axis=1).transpose())).transpose()[:, 2]

    # 得到雷达点云投影到像素平面的位置
    ip = cv2.projectPoints(entry, self.rvec, self.tvec, self.K_0, self.C_0)[0].reshape(-1, 2).astype(np.int32)

    # 判断投影点是否在图像内部
    inside = np.logical_and(np.logical_and(ip[:, 0] >= 0, ip[:, 0] < self.size[0]),
                            np.logical_and(ip[:, 1] >= 0, ip[:, 1] < self.size[1]))
    ip = ip[inside]
    dpt = np.array(dpt[inside]).flatten()
    # 将各个点的位置 [N,2] 加入队列
    self.queue.put(ip)
    if self.queue.full():
        # 队满，执行出队操作，将出队的点云中所有点对应的投影位置的值置为 nan
        ip_d = self.queue.get()
        self.depth[ip_d[:, 1], ip_d[:, 0]] = np.nan
    # TODO: 如果点云有遮挡关系，则测距测到前或后不确定  其实我也遇到了这个问题
    # 更新策略，将进队点云投影点的 z 值与原来做比较，取均值
    s = np.stack([self.depth[ip[:, 1], ip[:, 0]], dpt], axis=1)
    s = np.nanmedian(s, axis=1)
    self.depth[ip[:, 1], ip[:, 0]] = s
```

**对神经网络给出的装甲板中心点附近区域进行深度估计，从而估计敌方车辆的坐标。**

```python
def depth_detect_refine(self, r):
        """
        :param r: the bounding box of armor , format (x0,y0,w,h)

        :return: (x0,y0,z) x0,y0 是中心点在归一化相机平面的坐标前两位，z 为其对应在相机坐标系中的 z 坐标值
        """
        center = np.float32([r[0] + r[2] / 2, r[1] + r[3] / 2])
        # 采用以中心点为基准点扩大一倍的装甲板框，并设置 ROI 上界和下界，防止其超出像素平面范围
        # area = self.depth[int(max(0, center[1] - r[3])):int(min(center[1] + r[3], self.size[1] - 1)),
        #        int(max(center[0] - r[2], 0)):int(min(center[0] + r[2], self.size[0] - 1))]
        area = self.depth[int(r[1]):int(r[1] + r[3]), int(r[0]):int(r[0] + r[2])]
        z = np.nanmean(area) if not np.isnan(area).all() else np.nan  # 当对应 ROI 全为 nan，则直接返回为 nan

        return z

def detect_depth(self, rects):
    """
    :param rects: List of the armor bounding box with format (x0,y0,w,h)

    :return: an array, the first dimension is the amount of armors input, and the second is the location data (x0,y0,z)
    x0,y0 是中心点在归一化相机平面的坐标前两位，z 为其对应在相机坐标系中的 z 坐标值
    """
    if len(rects) == 0:
        return np.stack([], axis=0) * np.nan

    ops = []

    for rect in rects:
        ops.append(self.depth_detect_refine(rect))

    return np.stack(ops, axis=0)
```

#### 可能问题

点云积分方式比较简单，在点云密集的区域，点云更新不及时，导致车辆闪过时得出的深度实际上为车辆后方区域的深度。

---

### 备用方案：德劳内三角定位（主要在 `radar_detect/location_Delaunay.py` 中）

在初始化 `location_Delaunay` 类时从 `config` 文件中导入预先手动标注坐标的场景中的基点，如各处地形边缘上的角点。注意 Debug 时和比赛时选点等方面的不同。

```python
from config import Delaunary_points, cam_config
...
...
if cam_side in ["cam_left", "cam_right"]:
    t = Delaunary_points[int(debug)][cam_side]
    self.cam_rect = t[0]
    self.c_points = np.array(t[1]).reshape((-1, 3)) * 1000
    self.c_points = self.c_points[list(np.array(choose) - 1)]
    ...
```

调用 OpenCV 中的德劳内三角分割，将标注点分割成三角区域，方便后续定位。

```python
self.Dly = cv.Subdiv2D(self.cam_rect)
```

德劳内分割的目标是得到较“胖”的三角形，避免得到过于“瘦长的三角形”，原应用于图形渲染中，可以避免瘦长三角形的存在对渲染效率的影响。
考虑四个点的情况，有两种分割方式如下。

![德劳内示意图 1](https://pic2.zhimg.com/80/v2-1ececb07e635ecef46df14141319a199.webp)

德劳内分割将检测每个点是否在另外三个点的外接圆外部，确保分割结果为第一种。

![德劳内示意图 2](https://pic1.zhimg.com/80/v2-b6add54cfcf4bbc96670989b504259ec.webp)

![德劳内示意图 3](https://pic1.zhimg.com/80/v2-9469cca302bcc9d8fae3e0a38e4589ec.webp)

在这里使用德劳内分割的主要目的是尽可能使划分出来的三角区域在同一场景区域内，从而尽可能确保车的图像在三角区域内时，车的真实位置也在三角顶点的真实坐标确定的平面上。

然后对目标点匹配离目标点最近的三角区域，分在顶点上、在边上、在内部三种情况讨论，分别用 `_cal_pos_vertex`，`_cal_pos_edge` 和 `_cal_pos_triangle` 解决。

**其中，后两种情况的大致思路为用三个顶点的坐标（可以得到两条边）线性表示目标点的坐标，从而用同样的线性关系确认深度（假定了目标点在真实空间中在顶点所在平面上）。**

```python
def get_point_pos(self, l: np.ndarray, detect_type: int):
    w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
    if not self.init_ok or l.shape[0] != 4:
        pass
    else:
        if detect_type == 2:
            try:
                res = self.Dly.findNearest(tuple(l[1:3]))[1]
                w_point = self._cal_pos_vertex(res)
            except Exception as e:
                w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
                print(f"[ERROR] {e}")
        elif detect_type == 1:
            try:
                res = self.Dly.locate(tuple(l[1:3]))
                value = res[0]
                if value == cv.SUBDIV2D_PTLOC_ERROR:
                    return np.ndarray((3, 1), dtype=np.float64) * np.nan
                if value == cv.SUBDIV2D_PTLOC_INSIDE:
                    first_edge = res[1]
                    second_edge = self.Dly.getEdge(first_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                    third_edge = self.Dly.getEdge(second_edge, cv.SUBDIV2D_NEXT_AROUND_LEFT)
                    w_point = self._cal_pos_triangle(
                        np.array([self.Dly.edgeDst(first_edge)[1],
                                    self.Dly.edgeDst(second_edge)[1],
                                    self.Dly.edgeDst(third_edge)[1],
                                    ]),
                        l[1:3]
                    )
                if value == cv.SUBDIV2D_PTLOC_ON_EDGE:
                    first_edge = res[1]
                    w_point = self._cal_pos_edge(
                        np.array([self.Dly.edgeOrg(first_edge)[1],
                                    self.Dly.edgeDst(first_edge)[1]
                                    ]),
                        l[1:3]
                    )
                if value == cv.SUBDIV2D_PTLOC_VERTEX:
                    w_point = self._cal_pos_vertex(np.ndarray(self.cam_points[res[2]]))
            except Exception as e:
                w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
                print(f"[ERROR] {e}")
        else:
            w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
        if w_point.reshape(-1).shape[0] == 0:
            w_point = np.ndarray((3, 1), dtype=np.float64) * np.nan
    return w_point
```

#### 可能问题

用德劳内分割不能保证车和三角顶点在空间中一定处于同个平面。如果选择了三个点并不在某个坡上或某面墙上，车的深度是不确定的。

---

## 今年想做的改动

- 改动雷达点云的积分方式或深度图的更新方式，比如在一些区域提高新测量值的权重，并在刚开始积分对内点深度插值等，暂时还没想好具体怎么做，总之目标是让实时性更好一点。
- 其他方案的改动还没想好

---

## 今年想做的新方案：双目深度估计

### 标定

左相机作为主相机，按原方法标定。

#### 方案一

右相机也由原方法标定。

#### 方案二

利用比赛场景中的角点在左右相机中的位置（人工标点，8 对以上）标定左右相机的相对位置。

#### 方案三

左右相机分别用原方法标定后，用角点在左右相机中的位置验证标定效果。

### 深度估计

**当跑神经网络耗时较久或装甲板识别（包括序号颜色）错误率较高时，同时对左右相机图像跑神经网络会造成较大的计算负担，也会放大装甲板识别的错误率，则应选用搜索的方式。如果神经网络性能较好，直接选点进行匹配即可。**

#### 方案一：直接匹配

1. 左右相机输出分别跑装甲板识别的网络，利用装甲板序号进行区域匹配。

2. 取两图区域中的中心点，和 4 个四分之一矩形的中心点进行匹配（共 5 对点）。

3. 用三角测量求出 5 个深度估计值，取平均值作为装甲板深度估计值（需要装甲板识别比较准）。

#### 方案二：极线搜索

1. 只用左相机的图像跑神经网络，在装甲板区域提取 ORB 特征点。

2. 截取特征点在场景下的可能深度范围对应的右相机投影点的线段，在所有特征点对应的线段群上提取 ORB 特征点。

3. 对每个特征点，搜索其对应线段上是否有匹配的特征点。

4. 用匹配成功的特征点对，估计特征点深度。取平均值作为装甲板深度估计值。

5. 如果特征点提取和匹配耗时过长，可以考虑加大搜索步长。
