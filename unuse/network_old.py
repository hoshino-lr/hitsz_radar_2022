'''
default network class
给神经网络类的接口格式定义，神经网络具体需要自行添加
'''

import pytrt
import numpy as np
import argparse
import time
from pathlib import Path
import cv2
import torch
import sys
import random
from threading import Thread

sys.path.insert(-1, '/home/hoshino/CLionProjects/hitsz_radar/net/yolov5')

from yolov5.utils.general import non_max_suppression
from yolov5.models.experimental import attempt_load


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain1 = img1_shape[0] / img0_shape[0]  # gain  = old / new
    gain2 = img1_shape[1] / img0_shape[1]
    coords[:, [0, 2]] /= gain2
    coords[:, [1, 3]] /= gain1
    if type(coords) is not np.ndarray:
        coords[:, 0].clamp_(0, img0_shape[1])  # x1
        coords[:, 1].clamp_(0, img0_shape[0])  # y1
        coords[:, 2].clamp_(0, img0_shape[1])  # x2
        coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords


def jigsaw(det_points, img_src):
    step = 213
    jig_picture = np.zeros((640, 640, 3), dtype=np.uint8)
    count = 0
    for i in det_points:
        cut_img = img_src[int(i[1]):int(i[3]), int(i[0]):int(i[2])]
        cut_img = cv2.resize(cut_img, (213, 213))
        u = count % 3 * 213
        v = count // 3 * 213
        jig_picture[v:v + 213, u:u + 213] = cut_img.copy()
        count += 1
        if count == 9:
            break
    return jig_picture


def net2_process(net2_input, det_points, shape):
    net2_output = [[], [], [], [], [], [], [], [], []]
    if net2_input is None:
        pass
    else:
        for i in net2_input:
            count = i[0] // 213 + i[1] // 213 * 3
            i[0] %= 213
            i[1] %= 213
            i[2] = i[0] + i[2]
            i[3] = i[1] + i[3]
            net2_output[count].append(i)
        for count, i in enumerate(net2_output):
            if not len(i):
                continue
            else:
                i = np.array(i)
                confidences = i[:, 4]
                index = np.where(confidences == np.max(confidences))[0][0]
                scale = (
                int(det_points[count, 3] - det_points[count, 1]), int(det_points[count, 2] - det_points[count, 0]))
                i[:, :4] = scale_coords((213, 213), i[:, :4], scale).round()
                i = i[index, :]
                i[0] += det_points[count, 0]
                i[1] += det_points[count, 1]
                i[2] += det_points[count, 0]
                i[3] += det_points[count, 1]
                net2_output[count] = i.tolist()
    return net2_output


class Predictor(object):
    # 输入图片与输出结果
    img_src = []
    output = []
    name = ""
    img_show = True

    # net1参数
    _net1_confThreshold = 0.2
    _net1_nmsThreshold = 0.3
    net1_inpHeight = 640
    net1_inpWidth = 640
    net1_pt_file = '/home/hoshino/CLionProjects/hitsz_radar/resources/best.pt'
    net1_device = 'cuda:0'
    net1_classes_num = [0]
    # 不检测base
    net1_classes = ['car', 'watcher', 'base']

    # net2参数
    net2_confThreshold = 0.2
    net2_nmsThreshold = 0.3
    net2_inpHeight = 640
    net2_inpWidth = 640
    net2_box_num = 25200
    net2_onnx_file = "/home/hoshino/CLionProjects/hitsz_radar/resources/mbv3.onnx"
    net2_trt_file = "/home/hoshino/CLionProjects/hitsz_radar/resources/mbv3.engine"
    net2_classes = ['R1', 'B1', 'R2', 'B2', 'R3', 'B3', 'R4', 'B4', 'R5', 'B5', 'R6', 'B6',
                    'R7', 'B7', 'R10', 'B10', 'R11', 'B11', 'RE', 'BE']

    def __init__(self, _name):
        # 初始化函数

        # 线程初始化
        self.__thread = Thread(target=self.thread_detect)
        self.name = _name
        # net1初始化
        self._net1 = attempt_load(self.net1_pt_file, self.net1_device)
        # net2初始化
        self._net2 = pytrt.Trt()
        self._net2.CreateEngine(self.net2_onnx_file, self.net2_trt_file, 1, 1)
        self._net2.SetDevice(0)

    def pub_sub_init(self, pub, sub):
        # 管道初始化
        self.__pub = pub
        self.__sub = sub

    def thread_start(self):
        # 开启多线程
        self.__thread.setDaemon(True)
        self.__thread.start()

    def detect_cars(self, src):
        # 检测函数
        self.img_src = src.copy()
        # 图像预处理
        img = cv2.resize(src, (self.net1_inpHeight, self.net1_inpWidth), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(self.net1_device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        # net1推理
        pred = self._net1(img)[0]
        pred = non_max_suppression(pred, self._net1_confThreshold, self._net1_nmsThreshold,
                                   classes=self.net1_classes_num,
                                   agnostic=True)

        res = []
        for det in pred:
            if len(det):
                # get Jigsaw
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.img_src.shape).round()
                net2_img = jigsaw(det[:, :4], self.img_src.copy())
                # Rescale boxes from img_size to im0 size
                net2_output = self.detect_armor(net2_img)
                net2_output = net2_process(net2_output, det[:, :4], self.img_src.shape)
                for index, i in enumerate(det):
                    img1_anchor = [int(i[0]), int(i[1]), int(i[2]), int(i[3])]
                    if index < 9:
                        res.append([img1_anchor, i[4], i[5], net2_output[index]])

        if self.img_show:  # 画图
            self.net_show(res)

        return res,self.img_src.copy()

    def detect_armor(self, src):
        img = cv2.resize(src, (self.net2_inpHeight, self.net2_inpWidth), interpolation=cv2.INTER_LINEAR)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        img = img.flatten()
        self._net2.CopyFromHostToDevice(img, 0)
        self._net2.Forward()
        net_output = self._net2.CopyFromDeviceToHost(1)
        res = self.process(net_output)
        return res

    def process(self, output):
        # 第二个网络的处理
        classIds = []
        confidences = []
        bboxes = []
        ratio_w = 1
        ratio_h = 1
        class_num = len(self.net2_classes)
        output = output.flatten()
        # 拉平
        for i in range(0, self.net2_box_num):
            data_idx = i * (class_num + 4)
            obj_conf = float(output[data_idx + 4])
            # 只筛选可信度大于阈值的，为后面的NMS减轻计算压力
            if obj_conf > self.net2_confThreshold:
                max_id = output[data_idx + 5]
                classIds.append(max_id)
                confidences.append(obj_conf)
                centerX = int(output[data_idx] * ratio_w)
                centerY = int(output[data_idx + 1] * ratio_h)
                width = int(output[data_idx + 2] * ratio_w)
                height = int(output[data_idx + 3] * ratio_h)
                left = int(centerX - width / 2)
                top = int(centerY - height / 2)
                bboxes.append([left, top, width, height])

        # NMS筛选
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.net2_confThreshold, self.net2_nmsThreshold)

        # 筛选出置信度最大的
        """
        if indices:
            select_confidences = [confidences[i[0]] for i in indices]
            max_index = indices[select_confidences.index(max(select_confidences))][0]
            bbox = [bboxes[max_index][1], bboxes[max_index][0],
                    bboxes[max_index][1] + bboxes[max_index][3], bboxes[max_index][0] + bboxes[max_index][2]]
            res = [bbox, confidences[max_index], classIds[max_index]]
        """
        if len(indices):
            res = []
            for i in indices:
                if bboxes[i[0]][0] < 0:
                    bboxes[i[0]][0] = 0
                if bboxes[i[0]][1] < 0:
                    bboxes[i[0]][1] = 0
                res.append([bboxes[i[0]][0], bboxes[i[0]][1], bboxes[i[0]][2], bboxes[i[0]][3], confidences[i[0]],
                            classIds[i[0]]])
        else:
            res = None
        return res

    def net_show(self, res):
        color = (255, 255, 0)
        line_thickness = 3
        tl = line_thickness  # line/font thickness
        for i in res:
            if len(i[3]):
                label = f'{self.net1_classes[int(i[2])]} {i[1]:.2f}  {self.net2_classes[int(i[3][5])]}'
            else:
                label = f'{self.net1_classes[int(i[2])]} {i[1]:.2f} None'
            # Plots one bounding box on image img
            if len(i[3]):
                c1, c2 = (int(i[3][0]), int(i[3][1])), (int(i[3][2]), int(i[3][3]))
                cv2.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
            c1, c2 = (i[0][0], i[0][1]), (i[0][2], i[0][3])
            cv2.rectangle(self.img_src, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)

            if label:
                tf = max(tl - 1, 1)  # font thickness
                t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
                c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
                cv2.rectangle(self.img_src, c1, c2, color, -1, cv2.LINE_AA)  # filled
                cv2.putText(self.img_src, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf,
                            lineType=cv2.LINE_AA)

    def thread_detect(self):
        # 多线程接收写法
        try:
            while True:
                self.img_src = self.__sub.sub()
                if self.img_src is not None:
                    t1 = time_synchronized()
                    self.detect_cars(self.img_src)
                    t2 = time_synchronized()
                    print(f'{self.name}:检测总时间. ({t2 - t1:.3f}s)')
                    t1 = time_synchronized()
                else:
                    time.sleep(0.01)
                    print("sleep")
        except Exception as e:
            print(e)


if __name__ == '__main__':
    sys.path.append("..")
    from mul_manager.mul_manager import MulManager
    from yolov5.utils.torch_utils import time_synchronized

    from yolov5.utils.datasets import LoadImages
    from yolov5.utils.general import increment_path

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/hoshino/CLionProjects/hitsz_radar/resources/best.pt',
                        help='model.pt')
    parser.add_argument('--source', type=str,
                        default='/home/hoshino/CLionProjects/hitsz_radar/resources/two_cam/1.mp4',
                        help='source')  # file/folder, 0 for webcam
    parser.add_argument('--img-size', type=int, default=640, help='inference size (pixels)')
    parser.add_argument('--conf-thres', type=float, default=0.45, help='object confidence threshold')
    parser.add_argument('--iou-thres', type=float, default=0.3, help='IOU threshold for NMS')
    parser.add_argument('--device', default='1', help='cuda device, i.e. 0 or 0,1,2,3 or cpu')
    parser.add_argument('--view-img', action='store_true', help='display results')
    parser.add_argument('--save-txt', action='store_true', help='save results to *.txt')
    parser.add_argument('--save-conf', action='store_true', help='save confidences in --save-txt labels')
    parser.add_argument('--classes', nargs='+', type=int, default=[0, 1, 2],
                        help='filter by class: --class 0, or --class 0 2 3')
    parser.add_argument('--agnostic-nms', action='store_true', help='class-agnostic NMS')
    parser.add_argument('--augment', action='store_true', help='augmented inference')
    parser.add_argument('--project', default='runs/detect', help='save results to project/name')
    parser.add_argument('--name', default='exp', help='save results to project/name')
    parser.add_argument('--exist-ok', action='store_true', help='existing project/name ok, do not increment')
    opt = parser.parse_args()
    save_img = True
    save_txt = False
    vid_path, vid_writer = None, None
    # 使用了yolov5代码的一个函数，
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    dataset = LoadImages(opt.source, img_size=640)
    predictor1 = Predictor("nee1")
    predictor2 = Predictor("nee2")
    # time.sleep(10)
    # 多线程管理，不太完善
    # mul = MulManager()
    # predictor1.pub_sub_init(mul.create_pub("net1_pub"), mul.create_sub("net1_sub", 1))
    # predictor2.pub_sub_init(mul.create_pub("net2_pub"), mul.create_sub("net2_sub", 1))
    # pub2 = mul.create_pub("net2_sub")
    # pub1 = mul.create_pub("net1_sub")
    # predictor1.thread_start()
    # predictor2.thread_start()
    print("主线程开始工作")
    num = 0
    cv2.namedWindow("img_src",cv2.WINDOW_NORMAL)
    for path, img, im0s, vid_cap in dataset:
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        t1 = time_synchronized()
        _,picture = predictor1.detect_cars(im0)
        t2 = time_synchronized()
        print(f':检测总时间. ({t2 - t1:.3f}s)')
        cv2.imshow("img_src",picture)
        cv2.waitKey(2)
    cv2.destroyAllWindows()