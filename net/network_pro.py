''' 
网络类（适合多进程） 使用tensorrtx 接口
created by 李龙 in 2020/11
最终修改版本 李龙 2021/1/16
'''
import multiprocessing
import numpy as np
import time
import cv2
import os
from multiprocessing import Process
from resources.config import net1_onnx, net2_onnx, net1_engine, \
    net2_engine, net1_cls, net2_cls_names, net2_col_names, \
    enemy_color
from net.tensorrtx import YoLov5TRT
from radar_detect.common import armor_filter


class Predictor(object):
    """
    格式定义： [N,bbox(xyxy),conf,cls,bbox(xyxy),conf,cls,col,N]
    """
    # 输入图片与输出结果
    img_src = []
    output = []
    name = ""
    img_show = False

    # net1参数
    net1_confThreshold = 0.3
    net1_nmsThreshold = 0.45
    net1_inpHeight = 640
    net1_inpWidth = 640
    net1_onnx_file = net1_onnx
    net1_trt_file = net1_engine
    # 不检测base

    net1_grid = []
    net1_num_anchors = [3, 3, 3]
    net1_anchors = [[10, 13], [16, 30], [33, 23], [30, 61], [62, 45], [59, 119], [116, 90], [156, 198], [373, 326]]
    net1_strides = [8, 16, 32]

    # net2参数
    net2_confThreshold = 0.25
    net2_nmsThreshold = 0.2
    net2_inpHeight = 640
    net2_inpWidth = 640
    net2_box_num = 25200
    net2_onnx_file = net2_onnx
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
        # net1初始化
        self._net1 = YoLov5TRT(self.net1_trt_file)
        # net2初始化
        self._net2 = YoLov5TRT(self.net2_trt_file)

        self.name = _name  # 选择的相机是左还是右
        self.choose_type = 0  # 只选择检测cars类，不检测哨兵和基地
        self.enemy_color = not enemy_color
        for i in self.net1_strides:
            self.net1_grid.append([self.net1_num_anchors[0], int(self.net1_inpHeight / i), int(self.net1_inpWidth / i)])

        for i in self.net2_strides:
            self.net2_grid.append([self.net2_num_anchors[0], int(self.net2_inpHeight / i), int(self.net2_inpWidth / i)])

    def pub_sub_init(self, pub, cam):
        # 管道初始化
        """
        :param pub
        :param cam
        """
        self.pub = pub
        self.sub = cam

    def detect_cars(self, src):
        """
        :param src 3072x2048
        :param 
        """
        # 检测函数

        self.img_src = src.copy()
        # 图像预处理
        # start = time.time()
        img = cv2.resize(self.img_src, (self.net1_inpHeight, self.net1_inpWidth), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0

        net1_output = self._net1.infer(img, 1)[0]
        res = self.net1_process(net1_output)
        # self.net1_time += time.time() - start
        if res.shape[0] != 0:
            # get Jigsaw
            # start = time.time()
            res[:, :4] = self.scale_coords((640, 640), res[:, :4], self.img_src.shape).round()
            net2_img = self.jigsaw(res[:, :4], self.img_src)
            # self.jaw_time += time.time() - start
            # Rescale boxes from img_size to im0 size
            # start = time.time()
            net2_output = self.detect_armor(net2_img)
            net2_output = self.net2_output_process(net2_output, det_points=res[:, :4], shape=self.img_src.shape)
            res = np.concatenate([res, net2_output], axis=1)

        if self.img_show and res.shape != 0:  # 画图
            self.net_show(res)
        # self.net2_time += time.time() - start
        armor_filter(res)
        return res, self.img_src

    def detect_armor(self, src):
        """
        :param src 输入一张640x640的图片
        :param res 输出一个(N,7)的array 或是None
        """
        img = cv2.resize(src, (self.net2_inpHeight, self.net2_inpWidth), interpolation=cv2.INTER_LINEAR)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x640x640
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        net_output = self._net2.infer(img, 3)
        res = self.net2_process(net_output[0], net_output[1], net_output[2])
        return res

    def net1_process_sjtu(self, output):
        # 第一个网络的处理
        """
        :return res 输出为一个(N,6)的array 或为一个空array
        """
        classIds = []
        confidences = []
        bboxes = []
        output = output.reshape(-1, 26)
        choose = output[:, 4] > self.net1_confThreshold
        output = output[choose]
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
                c = choose[i] // 1600
                h = (choose[i] % 1600) // self.net1_grid[n][1]
                w = (choose[i] % 1600) % self.net1_grid[n][2]
            anchor = self.net1_anchors[n * self.net1_grid[n][0] + c]
            xc = output[i, :]
            max_id = np.argmax(xc[5:])  # 选择置信度最高的 class
            if max_id != self.choose_type:  # 只选择最高的
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
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.net1_confThreshold, self.net1_nmsThreshold)
        res = []

        if len(indices):
            for i in indices:
                # 暂时为完成 boxes 转 numpy 
                bbox = [float(x) for x in bboxes[i[0]]]
                bbox.append(confidences[i[0]])
                bbox.append(float(classIds[i[0]]))
                res.append(bbox)

        res = np.array(res)
        if res.shape[0] != 0:
            res[:, 2] = res[:, 0] + res[:, 2]
            res[:, 3] = res[:, 1] + res[:, 3]
        return res

    def net1_process(self, output):
        """
        # 第一个网络的处理
        :param output
        :return res 输出为一个(N,6)的array 或为一个空array
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
                c = choose[i] // 1600
                h = (choose[i] % 1600) // self.net1_grid[n][1]
                w = (choose[i] % 1600) % self.net1_grid[n][2]
            anchor = self.net1_anchors[n * self.net1_grid[n][0] + c]
            xc = output[i, :]
            max_id = np.argmax(xc[5:])  # 选择置信度最高的 class
            if max_id != self.choose_type:  # 只选择最高的
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
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.net1_confThreshold, self.net1_nmsThreshold)
        res = []

        if len(indices):
            for i in indices:
                # 暂时为完成 boxes 转 numpy
                bbox = [float(x) for x in bboxes[i[0]]]
                bbox.append(confidences[i[0]])
                bbox.append(float(classIds[i[0]]))
                res.append(bbox)

        res = np.array(res)
        if res.shape[0] != 0:
            res[:, 2] = res[:, 0] + res[:, 2]
            res[:, 3] = res[:, 1] + res[:, 3]
        return res

    @staticmethod
    def sigmoid(x):
        return 1 / (1 + np.exp(-x))

    def net2_process(self, output1, output2, output3):
        """
        :param 第一层 19200 输入
        :param 第二层 4800 输入
        :param 第三层 1200 输入
        """
        # 第二个网络的处理
        classIds = []
        colorIds = []
        confidences = []
        bboxes = []
        output1 = output1.reshape(-1, 28)  # 1x19200x28
        output2 = output2.reshape(-1, 28)  # 1x4800x28
        output3 = output3.reshape(-1, 28)  # 1x1600x28
        net_output = np.concatenate([np.concatenate([output1, output2], axis=0), output3], axis=0)
        choose = self.sigmoid(net_output[:, 4]) > self.net1_confThreshold
        output = net_output[choose]
        output = self.sigmoid(output)
        choose = np.where(choose == True)[0]
        for i in range(0, len(choose)):
            if choose[i] < 19200:
                n = 0
                c = choose[i] // 6400
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
            anchor = self.net2_anchors[n * self.net2_grid[n][0] + c]
            xc = output[i, :]
            cls_id = np.argmax(xc[15:15 + 9])  # 选择置信度最高的 class
            col_id = np.argmax(xc[24:])  # 选择置信度最高的 color
            if col_id != self.enemy_color and cls_id in range(1, 6):
                continue
            obj_conf = float(xc[4])  # 置信度
            centerX = int(
                (xc[0] * 2 - 0.5 + w) / self.net1_grid[n][2] * self.net2_inpWidth)
            centerY = int(
                (xc[1] * 2 - 0.5 + h) / self.net1_grid[n][1] * self.net2_inpHeight)
            width = int(pow(xc[2] * 2, 2) * anchor[0])
            height = int(pow(xc[3] * 2, 2) * anchor[1])
            left = int(centerX - width / 2)
            top = int(centerY - height / 2)
            bboxes.append([left, top, width, height])
            classIds.append(cls_id)
            colorIds.append(col_id)
            confidences.append(obj_conf)

        # NMS筛选
        indices = cv2.dnn.NMSBoxes(bboxes, confidences,
                                   self.net2_confThreshold, self.net2_nmsThreshold)

        if len(indices):
            res = []
            for i in indices:
                res.append([bboxes[i[0]][0], bboxes[i[0]][1], bboxes[i[0]][2],
                            bboxes[i[0]][3], confidences[i[0]],
                            classIds[i[0]], colorIds[i[0]]])
        else:
            res = None
        return res

    @staticmethod
    def scale_coords(img1_shape, coords, img0_shape):
        # Rescale coords (xyxy) from img1_shape to img0_shape
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
        # 拼图 用于第二层检测的前处理
        # 将每一辆车的图片resize成213x213的图片，放进9x9的网格图片中
        # 输出为一张640x640的图片
        step = 213
        jig_picture = np.zeros((640, 640, 3), dtype=np.uint8)
        count = 0
        for i in det_points:
            try:
                cut_img = img_src[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
                cut_img = cv2.resize(cut_img, (213, 213))
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
        #  对第二层网络输出的处理
        net2_boxes = [[], [], [], [], [], [], [], [], []]
        net2_output = np.full((det_points.shape[0], 8), np.nan)
        for i in range(det_points.shape[0]):  # 满足reporject类的需求，对装甲板进行编号
            net2_output[i, 7] = i
        if net2_input is None:
            pass
        else:
            for i in net2_input:
                count = i[0] // 213 + i[1] // 213 * 3
                i[0] %= 213
                i[1] %= 213
                i[2] = i[0] + i[2]
                i[3] = i[1] + i[3]
                net2_boxes[count].append(i)
            for count, i in enumerate(net2_boxes):
                if not len(i) or count >= det_points.shape[0]:
                    continue
                else:
                    i = np.array(i)
                    confidences = i[:, 4]
                    index = np.where(confidences == np.max(confidences))[0][0]
                    scale = (
                        int(det_points[count, 3] - det_points[count, 1]),
                        int(det_points[count, 2] - det_points[count, 0]))
                    i[:, :4] = self.scale_coords((213, 213), i[:, :4], scale).round()
                    i = i[index, :]
                    i[0] += det_points[count, 0]
                    i[1] += det_points[count, 1]
                    i[2] += det_points[count, 0]
                    i[3] += det_points[count, 1]
                    net2_output[count, 0:7] = i[:]
        return net2_output

    def stop(self):
        """
        停止ternsorrt线程，在关闭之前必须做这个操作，不然tensorrt的stramer可能无法释放
        """
        self._net1.destroy()
        self._net2.destroy()

    def net_show(self, res):
        # 绘制函数
        color = (255, 0, 255)
        line_thickness = 3
        tl = line_thickness  # line/font thickness
        for i in res:
            if not np.isnan(i[11]):
                label = f'{net1_cls[int(i[5])]} {i[4]:.2f}  {net2_cls_names[int(i[11])]}'
            else:
                label = f'{net1_cls[int(i[5])]} {i[4]:.2f} None'
            # Plots one bounding box on image img
            if not np.isnan(i[11]):
                c1, c2 = (int(i[6]), int(i[7])), (int(i[8]), int(i[9]))
                cv2.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            c1, c2 = (int(i[0]), int(i[1])), (int(i[2]), int(i[3]))
            cv2.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(self.img_src, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.img_src, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                            lineType=cv2.LINE_AA)


if __name__ == '__main__':
    import sys

    sys.path.append("..")  # 单独跑int的时候需要
    cap = cv2.VideoCapture("/home/hoshino/CLionProjects/LCR_sjtu/demo_resource/two_cam/1.mp4")
    PICS = []
    while cap.isOpened():
        res, frame = cap.read()
        if res:
            PICS.append(frame)
        else:
            break

    count = 0
    t2 = time.time()
    t1 = time.time()
    pre1 = Predictor('cam_left')

    for frame in PICS:
        _, pic = pre1.detect_cars(frame)
        count += 1
        pic = cv2.resize(pic, (1280, 720))
        # cv2.imshow("asd", pic)
        # cv2.waitKey(1)
        if time.time() - t1 > 1:
            print(f"fps:{count / (time.time() - t1)}")
            count = 0
            t1 = time.time()
    pre1.stop()
    print(time.time() - t2)
    print("主线程结束")
