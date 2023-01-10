"""
网络类（适合多进程） 使用tensorrtx 接口
created by 李龙 in 2020/11
最终修改版本 李龙 2021/1/16
添加注释 林顺喆 2022/12/26
"""
import numpy as np
import threading
import time
import cv2 as cv
from config import net1_engine, net2_engine, \
    net1_cls, net2_cls_names, enemy_color, cam_config
from net.tensorrtx import YoLov5TRT
from radar_detect.common import armor_filter

"""
"""
class Predictor(object):
    """
    格式定义： [N,  [bbox(xyxy),conf,cls,bbox(xyxy),conf,cls,col,N]
    """
    # 输入图片与输出结果
    output = []
    name = ""
    img_show = True
    record_state = False

    # net1参数
    net1_confThreshold = 0.4    # 初步筛选的置信度
    net1_nmsThreshold = 0.4     # NMS筛选参数
    net1_inpHeight = 640        # 输入网络的规范尺寸-高
    net1_inpWidth = 640         # 规范尺寸-宽
    net1_trt_file = net1_engine # TensorRT模型
    # 不检测base

    net1_grid = []              # 记录三层grid的划分信息
    net1_num_anchors = [3, 3, 3]    
    # anchor：对对象检测框的形状进行先验约束
    net1_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    net1_strides = [8, 16, 32]  # 三层grid划分的步长

    # net2参数
    net2_confThreshold = 0.7
    net2_nmsThreshold = 0.3
    net2_inpHeight = 640
    net2_inpWidth = 640
    net2_box_num = 25200
    net2_trt_file = net2_engine

    net2_grid = []
    net2_num_anchors = [3, 3, 3]
    net2_anchors = [[4, 5], [8, 10], [13, 16], [23, 29], [43, 55], [73, 105], [146, 217], [231, 300], [335, 433]]
    net2_strides = [8, 16, 32]

    def __init__(self, _name):
        """
        初始化函数
        :param _name 
        """
        self.using_net = cam_config[_name]["using_net"] # R:False L:True
        if self.using_net:
            # net1初始化
            self._net1 = YoLov5TRT(self.net1_trt_file)
            # net2初始化
            self._net2 = YoLov5TRT(self.net2_trt_file)
        self.img_src = np.zeros(cam_config[_name]["size"])  # 生成3072x2048的零矩阵（图片大小）
        self.name = _name  # 选择的相机是左还是右
        self.choose_type = 0  # 只选择检测cars类，不检测哨兵和基地
        self.enemy_color = not enemy_color
        self.lock = threading.Condition()   # 多线程操作

        # 划分三层grid
        for i in self.net1_strides:
            self.net1_grid.append([self.net1_num_anchors[0], int(self.net1_inpHeight / i), int(self.net1_inpWidth / i)])

        for i in self.net2_strides:
            self.net2_grid.append([self.net2_num_anchors[0], int(self.net2_inpHeight / i), int(self.net2_inpWidth / i)])

    def pub_sub_init(self, pub, cam):
        """
        管道初始化
        :param pub
        :param cam
        """
        self.pub = pub
        self.sub = cam

    def detect_cars(self, src):
        """
        检测函数
        :param src 3072x2048
        """
        if not self.using_net:
            return np.array([]), src
        else:
            self.lock.acquire()
            self.img_src = src.copy()
            self.lock.notify()
            self.lock.release()
            # 图像预处理
            img = cv.resize(self.img_src, (self.net1_inpHeight, self.net1_inpWidth), interpolation=cv.INTER_LINEAR)
            # 将图片通道顺序从BGR转到RGB
            img = img[:, :, ::-1].transpose(2, 0, 1)    # ::-1相当于于反转
            # 在内存中连续存储，提高访问速度
            img = np.ascontiguousarray(img) 
            img = img.astype(np.float32)    # 转换数值类型为float32
            img /= 255.0    

            # 把图片输入网络进行推理
            net1_output = self._net1.infer(img, 1)[0]
            res = self.net1_process_sjtu(net1_output)

            if res.shape[0] != 0:
                # 将第一层网络得到的640x640图片尺寸下的一组识别框的坐标转化为原图中的
                res[:, :4] = self.scale_coords((640, 640), res[:, :4], self.img_src.shape).round()
                # 拼图
                net2_img = self.jigsaw(res[:, :4], self.img_src)
                # 输入第二层网络进行推理和NMS筛选
                net2_output = self.detect_armor(net2_img)
                # 对网络输出结果进行处理
                net2_output = self.net2_output_process(net2_output, det_points=res[:, :4], shape=self.img_src.shape)
                # 横向拼接res和net2_output
                res = np.concatenate([res, net2_output], axis=1)

            # 画图
            if self.img_show and res.shape != 0:  
                self.net_show(res)
            res = armor_filter(res)
            return res, self.img_src

    # 第二层网络的推理
    def detect_armor(self, src):
        """
        :param src 输入一张640x640的图片
        Returns:
            res 输出一个(N,7)的array 或是None
        """
        img = cv.resize(src, (self.net2_inpHeight, self.net2_inpWidth), interpolation=cv.INTER_LINEAR)
        # 将图片的通道顺序从BGR转到RGB
        img = img[:, :, ::-1].transpose(2, 0, 1)
        # 连续存储
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        net_output = self._net2.infer(img, 3)
        # 后处理
        res = self.net2_process(net_output[0], net_output[1], net_output[2])
        return res

    def net1_process_sjtu(self, output):
        """
        第一个网络的处理
        :param output: 输入YoLov5TRT.infer方法的输出
        :return res: 输出为一个(N,6)的array 或为一个空array
        """
        classIds = []
        confidences = []
        bboxes = []
        # 将output数组reshape为(N,26)的矩阵
        output = output.reshape(-1, 26)
        # 每个框的置信度是否大于confThreshold（先这么理解）
        choose = output[:, 4] > self.net1_confThreshold
        # 对output所有元素取sigmond函数值
        output = self.sigmoid(output)
        # 这里应该是过滤掉所有置信度低于阈值的output
        output = output[choose]
        # choose数组复用，按顺序存储output中满足置信度条件的元素，在过滤前的output中的下标
        choose = np.where(choose == True)[0]
        for i in range(0, len(choose)):

            # 讨论划分grid的层别
            if choose[i] < 19200:   # 步长为8
                # 层编号
                n = 0
                # 推导框的anchor类型
                c = choose[i] // 6400
                # 计算检测框中心点所在grid cell的行列序号
                h = (choose[i] % 6400) // self.net1_grid[n][1]
                w = (choose[i] % 6400) % self.net1_grid[n][2]
            elif choose[i] > 24000: # 步长为32
                choose[i] = choose[i] - 24000
                n = 2
                c = choose[i] // 400
                h = (choose[i] % 400) // self.net1_grid[n][1]
                w = (choose[i] % 400) % self.net1_grid[n][2]
            else:                   # 步长为16
                choose[i] = choose[i] - 19200
                n = 1
                c = choose[i] // 1600
                h = (choose[i] % 1600) // self.net1_grid[n][1]
                w = (choose[i] % 1600) % self.net1_grid[n][2]
            anchor = self.net1_anchors[n * self.net1_grid[n][0] + c]
            xc = output[i, :]
            # 选择置信度最高的 class，认为是该框的class
            max_id = np.argmax(xc[5:]) 
            # 如果不是待检测的class，则不使用这一个框
            if max_id != self.choose_type:  
                continue
            obj_conf = float(xc[4] * xc[5 + max_id])  # 置信度
            # 计算检测框中心点在图片中的坐标
            centerX = int(
                (xc[0] * 2 - 0.5 + w) / self.net1_grid[n][2] * self.net2_inpWidth)
            centerY = int(
                (xc[1] * 2 - 0.5 + h) / self.net1_grid[n][1] * self.net2_inpHeight)
            # 计算检测框在图片中的width和height（像素）
            width = int(pow(xc[2] * 2, 2) * anchor[0])
            height = int(pow(xc[3] * 2, 2) * anchor[1])
            # 计算框的左上角的坐标
            left = int(centerX - width / 2)
            top = int(centerY - height / 2)

            #加入数组，以待进行NMS筛选
            bboxes.append([left, top, width, height])
            classIds.append(max_id)
            confidences.append(obj_conf)

        # NMS筛选，返回符合条件的框的下标数组
        indices = cv.dnn.NMSBoxes(bboxes, confidences, self.net1_confThreshold, self.net1_nmsThreshold)
        res = []

        if len(indices):
            # 变成(N,1)数组
            indices = indices.reshape(-1, 1)
            for i in indices:
                # 暂时为完成 boxes 转 numpy 
                bbox = [float(x) for x in bboxes[i[0]]]
                bbox.append(confidences[i[0]])
                bbox.append(float(classIds[i[0]]))
                res.append(bbox)

        res = np.array(res)
        if res.shape[0] != 0:
            # 计算右下角坐标，并替代宽高作为结果输出
            res[:, 2] = res[:, 0] + res[:, 2]
            res[:, 3] = res[:, 1] + res[:, 3]
        return res

    #
    def net1_process(self, output):
        """
        # 第一个网络的处理
        :param output
        :return res: 输出为一个(N,6)的array 或为一个空array
        """
        classIds = []
        confidences = []
        bboxes = []
        output = output.reshape(-1, 8)
        choose = self.sigmoid(output[:, 4]) > self.net1_confThreshold
        output = output[choose]
        output = self.sigmoid(output)
        choose = np.where(choose == True)[0]
        for i in range(0, len(choose)):
            if choose[i] < 19200:
                n = 0
                c = choose[i] // 6400
                h = (choose[i] % 6400) // self.net1_grid[n][1]
                w = (choose[i] % 6400) % self.net1_grid[n][2]
            elif choose[i] > 24000:
                choose[i] = choose[i] - 24000
                n = 2
                c = choose[i] // 400
                h = (choose[i] % 400) // self.net1_grid[n][1]
                w = (choose[i] % 400) % self.net1_grid[n][2]
            else:
                choose[i] = choose[i] - 19200
                n = 1
                c = choose[i] /0, 1/ 1600
                h = (choose[i] % 1600) // self.net1_grid[n][1]
                w = (choose[i] % 1600) % self.net1_grid[n][2]
            anchor = self.net1_anchors[n * self.net1_grid[n][0] + c]
            xc = output[i, :]
            # 选择置信度最高的 class，认为是该框的class
            max_id = np.argmax(xc[5:])  
            # 如果不是待检测的class，则不使用这一个框
            if max_id != self.choose_type:  
                continue
            obj_conf = float(xc[4] * xc[5 + max_id])  # 置信度
            centerX = int(
                (xc[0] * 2 - 0.5 + w) / self.net1_grid[n][2] * self.net2_inpWidth)
            centerY = int(
                (xc[1] * 2 - 0.5 + h) / self.net1_grid[n][1] * self.net2_inpHeight)
            width = int(pow(xc[2] * 2, 2) * anchor[0])
            height = int(pow(xc[3] * 2, 2) * anchor[1])
            left = int(centerX - width / 2)
            top = int(centerY - height / 2)
            bboxes.append([left, top, width, height])
            classIds.append(max_id)
            confidences.append(obj_conf)

        # NMS筛选
        indices = cv.dnn.NMSBoxes(bboxes, confidences, self.net1_confThreshold, self.net1_nmsThreshold)
        res = []

        if len(indices):
            indices = indices.reshape(-1, 1)
            for i in indices:
                # 暂时为完成 boxes 转 numpy
                bbox = [float(x) for x in bboxes[i[0]]]
                bbox.append(confidences[i[0]])
                bbox.append(float(classIds[i[0]]))
                res.append(bbox)

        res = np.array(res)
        if res.shape[0] != 0:
            res[:, 2] = res[:, 0, 10] + res[:, 2]
            res[:, 3] = res[:, 1] + res[:, 3]
        return res

    # sigmond函数
    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def net2_process(self, output1, output2, output3):
        """
        第二个网络的处理
        :param output1 第一层 19200 输入
        :param output2 第二层 4800 输入
        :param output3 第三层 1200 输入
        :return: 输出一个(N,7)的ndarray
        """

        classIds = []
        colorIds = []
        confidences = []
        bboxes = []
        output1 = output1.reshape(-1, 28)  # 1x19200x28
        output2 = output2.reshape(-1, 28)  # 1x4800x28
        output3 = output3.reshape(-1, 28)  # 1x1600x28
        # 将3层grid的结果合并
        net_output = np.concatenate([np.concatenate([output1, output2], axis=0), output3], axis=0)
        # 过滤不满足置信度要求的检测框
        choose = self.sigmoid(net_output[:, 4]) > self.net1_confThreshold
        output = net_output[choose]
        output = self.sigmoid(output)
        choose = np.where(choose == True)[0]
        for i in range(0, len(choose)):
            if choose[i] < 19200:
                # 层编号
                n = 0
                # 推导该框的anchor类型
                c = choose[i] // 6400
                # 计算检测框中心点位置所在的grid cell
                h = (choose[i] % 6400) // self.net2_grid[n][1]
                w = (choose[i] % 6400) % self.net2_grid[n][2]
            elif choose[i] > 24000:
                choose[i] = choose[i] - 24000
                n = 2
                c = choose[i] // 400
                h = (choose[i] % 400) // self.net2_grid[n][1]
                w = (choose[i] % 400) % self.net2_grid[n][2]
            else:
                choose[i] = choose[i] - 19200
                n = 1
                c = choose[i] // 1600
                h = (choose[i] % 1600) // self.net2_grid[n][1]
                w = (choose[i] % 1600) % self.net2_grid[n][2]
            # 确定检测框anchor类型
            anchor = self.net2_anchors[n * self.net2_grid[n][0] + c]
            xc = output[i, :]
            cls_id = np.argmax(xc[15:15 + 9])  # 选择置信度最高的 class
            col_id = np.argmax(xc[24:])  # 选择置信度最高的 color
            if col_id != self.enemy_color or cls_id not in range(1, 6):
                continue
            obj_conf = float(xc[4])  # 置信度
            # 计算检测框中心点在图片中的坐标
            centerX = int(
                (xc[0] * 2 - 0.5 + w) / self.net1_grid[n][2] * self.net2_inpWidth)
            centerY = int(
                (xc[1] * 2 - 0.5 + h) / self.net1_grid[n][1] * self.net2_inpHeight)
            # 计算检测框在图片中的width和height（像素）
            width = int(pow(xc[2] * 2, 2) * anchor[0])
            height = int(pow(xc[3] * 2, 2) * anchor[1])
            # 计算左上角坐标
            left = int(centerX - width / 2)
            top = int(centerY - height / 2)
            bboxes.append([left, top, width, height])
            classIds.append(cls_id)
            colorIds.append(col_id)
            confidences.append(obj_conf)

        # NMS筛选
        indices = cv.dnn.NMSBoxes(bboxes, confidences,
                                  self.net2_confThreshold, self.net2_nmsThreshold)

        if len(indices):
            res = []
            indices = indices.reshape(-1, 1)
            for i in indices:
                # 0-左上角横坐标；1-左上角纵坐标；2-框宽；3-框高；4-置信度；5-小车class（与编号对应）；6-小车颜色（阵营）
                res.append([bboxes[i[0]][0], bboxes[i[0]][1], bboxes[i[0]][2],
                            bboxes[i[0]][3], confidences[i[0]],
                            classIds[i[0]], colorIds[i[0]]])
        else:
            res = None
        return res

    @staticmethod
    def scale_coords(img1_shape, coords, img0_shape):
        """
        将coord中所有框的坐标信息从 img1_shape图片尺寸下的坐标 放缩为 img0_shape尺寸下的坐标
        :param img1_shape:二元组，表示原图宽高（像素）
        :param coords: (N,4)的ndarray类型
        :param img0_shape: 二元组，表示目标图宽高（像素）
        :return: 放缩修改后的coords
        """
        gain1 = img1_shape[0] / img0_shape[0]  # gain  = old / new
        gain2 = img1_shape[1] / img0_shape[1]
        coords[:, [0, 2]] /= gain2
        coords[:, [1, 3]] /= gain1
        coords[:, 0] = np.clip(coords[:, 0], 0, img0_shape[1])  # x1
        coords[:, 1] = np.clip(coords[:, 1], 0, img0_shape[0])  # y1
        coords[:, 2] = np.clip(coords[:, 2], 0, img0_shape[1])  # x2
        coords[:, 3] = np.clip(coords[:, 3], 0, img0_shape[0])  # y2
        return coords

    @staticmethod
    def jigsaw(det_points, img_src):
        """
        拼图 用于第二层检测的预处理
        裁切图中每个检测框框住的部分，resize成213x213的图片，放进3x3的九宫格图片中
        :param det_points: (N,4) ndarray类型，给出检测框的左上角和右下角的横纵坐标
        :param img_src: 待裁切图片
        :return: 640x640像素的九宫格拼图
        """

        jig_picture = np.zeros((640, 640, 3), dtype=np.uint8)
        count = 0
        for i in det_points:
            try:
                cut_img = img_src[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                cut_img = cv.resize(cut_img, (213, 213))
                u = count % 3 * 213
                v = count // 3 * 213
                jig_picture[v:v + 213, u:u + 213] = cut_img.copy()
                count += 1
                if count == 9:
                    break
            except:
                print("error!")
        return jig_picture

    def net2_output_process(self, net2_input, det_points, shape):
        """
        对第二层网络输出的二次处理
        :param net2_input: 输入第二层网络初步处理（于net2_process中）后的网络输出
        :param det_points: 输入第一层网络处理后的网络输出（检测框信息），顺序与九宫格拼图对应
        :return: 所有九宫格的检测框信息
        """
        net2_boxes = [[], [], [], [], [], [], [], [], []]
        # 生成det_points长度x8，值初始化为nan的矩阵
        net2_output = np.full((det_points.shape[0], 8), np.nan)
        for i in range(det_points.shape[0]):  # 满足reporject类的需求，对装甲板进行编号
            net2_output[i, 7] = i
        if net2_input is None:
            pass
        else:
            for i in net2_input: 
                # 用网络得到的识别框的左上角 坐标，推导其所在九宫格的编号（顺序从上到下，从左到右）
                count = i[0] // 213 + (i[1] // 213) * 3
                # 得到在九宫格一格中的相对坐标
                i[0] %= 213
                i[1] %= 213
                # 计算右下角坐标
                i[2] = i[0] + i[2]
                i[3] = i[1] + i[3]

                # 将框的信息存入对应九宫格的列表
                net2_boxes[count].append(i)
            for count, i in enumerate(net2_boxes):
                if not len(i) or count >= det_points.shape[0]:
                    continue
                else:
                    # i是九宫格一格内所有识别框信息的数组
                    i = np.array(i)
                    # 每个框的置信度数组
                    confidences = i[:, 4]
                    # 九宫格每一格只选取一格置信度最高的框
                    index = np.where(confidences == np.max(confidences))[0][0]
                    scale = (
                        int(det_points[count, 3] - det_points[count, 1]),
                        int(det_points[count, 2] - det_points[count, 0]))
                    # 将框的坐标从213x213下的坐标放缩为原图切片下的坐标
                    i[:, :4] = self.scale_coords((213, 213), i[:, :4], scale).round()
                    i = i[index, :]
                    # 再转化为原图坐标
                    i[0] += det_points[count, 0]
                    i[1] += det_points[count, 1]
                    i[2] += det_points[count, 0]
                    i[3] += det_points[count, 1]
                    net2_output[count, 0:7] = i[:]
        return net2_output

    def stop(self):
        """
        停止tensorrt线程，在关闭之前必须做这个操作，不然tensorrt的streamer可能无法释放
        """
        if self.using_net:
            del self._net1
            del self._net2

    def net_show(self, res):
        """绘制函数"""
        color = (255, 0, 255)
        line_thickness = 3
        # 线条和字体的thickness
        tl = line_thickness  
        for i in res:
            # 连接前res为 (N,6) ，net2_output为 (N,8)，i[11]为小车类型（对应编号）
            if not np.isnan(i[11]):
                # 类型（car/watcher/base）、置信度、小车编号
                label = f'{net1_cls[int(i[5])]} {i[4]:.2f}  {net2_cls_names[int(i[11])]}'
            else:
                label = f'{net1_cls[int(i[5])]} {i[4]:.2f} None'
            # Plots one bounding box on image img
            # 再输出图像上画一个框
            if not np.isnan(i[11]):
                c1, c2 = (int(i[6]), int(i[7])), (int(i[8]), int(i[9]))
                cv.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)
            c1, c2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
            cv.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv.LINE_AA)

            # 在一旁添加文字说明
            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv.rectangle(self.img_src, c1, c2, color, -1, cv.LINE_AA)  # filled
                cv.putText(self.img_src, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                           lineType=cv.LINE_AA)


# TODO: 去掉无用的代码
if __name__ == '__main__':
    import sys

    sys.path.append("..")  # 单独跑int的时候需要
    cap = cv.VideoCapture("/home/hoshino/CLionProjects/hitsz_radar/resources/records/radar_data/19_13_36_left.avi")

    count = 0
    t2 = time.time()
    t1 = time.time()
    pre1 = Predictor('cam_left')

    while cap.isOpened():
        res, frame = cap.read()
        if not res:
            break
        _, pic = pre1.detect_cars(frame)
        count += 1
        pic = cv.resize(pic, (1280, 720))
        cv.imshow("asd", pic)
        res = cv.waitKey(1)
        if res == ord('q'):
            break
        if time.time() - t1 > 1:
            print(f"fps:{count / (time.time() - t1)}")
            count = 0
            t1 = time.time()
    pre1.stop()
    print(time.time() - t2)
    print("主线程结束")
