'''
default network class
给神经网络类的接口格式定义，神经网络具体需要自行添加
'''

import pickle as pkl
from ctypes import *
import pytrt
import cv2
import numpy as np
import argparse
import time
from pathlib import Path
import cv2
import torch
import sys

sys.path.insert(-1, '/home/hoshino/CLionProjects/hitsz_radar/net/yolov5')
sys.path.append("..")
from mul_manager.mul_manager import MulManager
from yolov5.utils.general import non_max_suppression
import random
import threading


def scale_coords(img1_shape, coords, img0_shape):
    # Rescale coords (xyxy) from img1_shape to img0_shape
    gain1 = img1_shape[0] / img0_shape[0]  # gain  = old / new
    gain2 = img1_shape[1] / img0_shape[1]
    pad = (img1_shape[1] - img0_shape[1] * gain2) / 2, (img1_shape[0] - img0_shape[0] * gain1) / 2  # wh padding
    coords[:, [0, 2]] -= pad[0]  # x padding
    coords[:, [1, 3]] -= pad[1]  # y padding
    coords[:, [0, 2]] /= gain2
    coords[:, [1, 3]] /= gain1
    coords[:, 0].clamp_(0, img0_shape[1])  # x1
    coords[:, 1].clamp_(0, img0_shape[0])  # y1
    coords[:, 2].clamp_(0, img0_shape[1])  # x2
    coords[:, 3].clamp_(0, img0_shape[0])  # y2
    return coords


def plot_one_box(x, img, color=None, label=None, line_thickness=None):
    # Plots one bounding box on image img
    tl = line_thickness or round(0.002 * (img.shape[0] + img.shape[1]) / 2) + 1  # line/font thickness
    color = color or [random.randint(0, 255) for _ in range(3)]
    c1, c2 = (int(x[0]), int(x[1])), (int(x[2]), int(x[3]))
    cv2.rectangle(img, c1, c2, color, thickness=tl, lineType=cv2.LINE_AA)
    if label:
        tf = max(tl - 1, 1)  # font thickness
        t_size = cv2.getTextSize(label, 0, fontScale=tl / 3, thickness=tf)[0]
        c2 = c1[0] + t_size[0], c1[1] - t_size[1] - 3
        cv2.rectangle(img, c1, c2, color, -1, cv2.LINE_AA)  # filled
        cv2.putText(img, label, (c1[0], c1[1] - 2), 0, tl / 3, [225, 255, 255], thickness=tf, lineType=cv2.LINE_AA)


class Predictor(object):
    img_src = []
    output = []

    __sub_num = 3
    _net1_confThreshold = 0.2
    _net1_nmsThreshold = 0.3
    net1_inpHeight = 640
    net1_inpWidth = 640
    net1_pt_file = '/home/hoshino/CLionProjects/hitsz_radar/resources/best.pt'
    net1_device = 'cuda:0'
    net1_classes_num = [0, 1, 2]
    net1_classes = ['car', 'watcher', 'base']

    net2_confThreshold = 0.3
    net2_nmsThreshold = 0.1
    net2_inpHeight = 640
    net2_inpWidth = 640
    net2_box_num = 25200
    net2_onnx_file = "/home/hoshino/CLionProjects/hitsz_radar/resources/mbv3.onnx"
    net2_trt_file = "/home/hoshino/CLionProjects/hitsz_radar/resources/mbv3.engine"
    net2_classes = ['R1', 'B1', 'R2', 'B2', 'R3', 'B3', 'R4', 'B4', 'R5', 'B5', 'R6', 'B6',
                    'R7', 'B7', 'R10', 'B10', 'R11', 'B11', 'RE', 'BE']

    def __init__(self):
        self.__thread = threading.Thread(target=self.detect)
        self._net2 = pytrt.Trt()
        # self._net2.CreateEngine(self.net2_onnx_file, self.net2_trt_file, [], 1, 1, [[]])
        self._net2.CreateEngine(self.net2_onnx_file, self.net2_trt_file, 1, 1)
        self._net1 = attempt_load(self.net1_pt_file, map_location=self.net1_device)

    def pub_sub_init(self, pub, sub):
        self.__pub = pub
        self.__sub = sub

    def thread_start(self):
        self.__thread.setDaemon(True)
        self.__thread.start()

    def detect_cars(self, src):
        self.img_src = src.copy()
        img = cv2.resize(src, (self.net2_inpHeight, self.net2_inpWidth), interpolation=cv2.INTER_LINEAR)
        img = img[:, :, ::-1].transpose(2, 0, 1)
        img = np.ascontiguousarray(img)
        img = torch.from_numpy(img).float().to(self.net1_device)
        img /= 255.0
        if img.ndimension() == 3:
            img = img.unsqueeze(0)
        pred = self._net1(img)[0]
        pred = non_max_suppression(pred, self._net1_confThreshold, self._net1_nmsThreshold,
                                   classes=self.net1_classes_num,
                                   agnostic=True)
        srcshow = self.img_src.copy()
        for i, det in enumerate(pred):
            if len(det):
                # Rescale boxes from img_size to im0 size
                det[:, :4] = scale_coords(img.shape[2:], det[:, :4], self.img_src.shape).round()
                for *xyxy, conf, cls1 in reversed(det):
                    img2 = self.img_src.copy()[int(xyxy[1]):int(xyxy[3]), int(xyxy[0]):int(xyxy[2])]
                    max_con = self.detect_armor(img2)
                    save_img = True
                    if save_img:  # Add bbox to image
                        label = f'{self.net1_classes[int(cls1)]} {conf:.2f} \t {self.net2_classes[int(max_con)]}'
                        plot_one_box(xyxy, srcshow, label=label, color=(255, 255, 0), line_thickness=3)
        return srcshow

    def detect_armor(self, src):
        img = cv2.resize(src, (self.net2_inpHeight, self.net2_inpWidth), interpolation=cv2.INTER_LINEAR)
        # Convert
        img = img[:, :, ::-1].transpose(2, 0, 1)  # BGR to RGB, to 3x416x416
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img /= 255.0
        img = img.flatten()

        # net_output = self._net2.detect(img.tolist(), self.net2_inpHeight, self.net2_inpWidth)
        self._net2.CopyFromHostToDevice(img, 0)
        self._net2.Forward()
        net_output = self._net2.CopyFromDeviceToHost(1)
        res = self.process(net_output)
        return res

    def process(self, output):
        classIds = []
        confidences = []
        bboxes = []
        ratio_w = 1
        ratio_h = 1
        class_num = len(self.net2_classes)
        output = output.flatten()
        for i in range(0, self.net2_box_num):
            data_idx = i * (class_num + 4)
            obj_conf = float(output[data_idx + 4])
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
        res = []
        indices = cv2.dnn.NMSBoxes(bboxes, confidences, self.net2_confThreshold, self.net2_nmsThreshold)
        for i in indices:
            box = bboxes[i[0]]
            res.append([box, classIds[i[0]], confidences[i[0]]])
        confidences = [x[2] for x in res]
        if confidences:
            max_con = res[confidences.index(max(confidences))][1]
        else:
            max_con = -1
        return max_con

    def net_show(self):
        pass

    def detect(self):
        try:
            t1 = time_synchronized()
            num = 0
            while 1:
                self.img_src = self.__sub.sub()
                if self.img_src is not None:
                    self.output = self.detect_cars(self.img_src)
                    t2 = time_synchronized()
                    print(f'detect. ({t2 - t1:.3f}s) num :{num}')
                    t1 = time_synchronized()
                    num = 0
                else:
                    num += 1
                    time.sleep(0.001)

        except Exception as e:
            print(e)


if __name__ == '__main__':
    from yolov5.utils.torch_utils import time_synchronized
    from yolov5.models.experimental import attempt_load
    from yolov5.utils.datasets import LoadImages
    from yolov5.utils.general import increment_path
    from concurrent.futures import ThreadPoolExecutor

    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', nargs='+', type=str,
                        default='/home/hoshino/CLionProjects/hitsz_radar/resources/best.pt',
                        help='model.pt path(s)')
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
    save_dir = Path(increment_path(Path(opt.project) / opt.name, exist_ok=opt.exist_ok))  # increment run
    (save_dir / 'labels' if save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir
    dataset = LoadImages(opt.source, img_size=640)
    predictor1 = Predictor()
    predictor2 = Predictor()
    mul = MulManager()
    predictor1.pub_sub_init(mul.create_pub("net1_pub"), mul.create_sub("net1_sub", 1))
    predictor2.pub_sub_init(mul.create_pub("net2_pub"), mul.create_sub("net2_sub", 1))
    pub2 = mul.create_pub("net2_sub")
    pub1 = mul.create_pub("net1_sub")
    predictor1.thread_start()
    predictor2.thread_start()
    for path, img, im0s, vid_cap in dataset:
        p, s, im0, frame = path, '', im0s, getattr(dataset, 'frame', 0)
        p = Path(p)
        save_path = str(save_dir / p.name)
        im1s = im0s.copy()
        pub1.pub(im0s)
        pub2.pub(im1s)
        """
        if save_img:
            if dataset.mode == 'image':
                cv2.imwrite(save_path, future1.result())
            else:  # 'video'
                if vid_path != save_path:  # new video
                    vid_path = save_path
                    if isinstance(vid_writer, cv2.VideoWriter):
                        vid_writer.release()  # release previous video writer
                    fourcc = 'mp4v'  # output video codec
                    fps = vid_cap.get(cv2.CAP_PROP_FPS)
                    w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                    h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                    vid_writer = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*fourcc), fps, (w, h))
                vid_writer.write(output)
                """
