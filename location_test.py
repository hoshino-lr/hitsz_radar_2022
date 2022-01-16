"""
雷达主程序
"""

from camera.cam import Camera
from net.network_pro import Predictor
from resources.config import DEBUG, USEABLE
from mapping.mainEntry import Mywindow
from radar_detect.location_alarm import Alarm

from PyQt5.QtCore import QTimer
from PyQt5 import QtWidgets
import logging
import cv2 as cv
import numpy as np
import sys


def spin_once():
    pic_left = None
    pic_right = None
    res_left, res_right = None,None
    if USEABLE['cam_left']:
        frame = cam.get_img()
        if frame is not None:
            im1 = frame.copy()
            res_left,frame = Predictor1.detect_cars(frame)
    if res_left is None:
        pass
    else:
        armors = res_left[:, [11, 6, 7, 8, 9]]
        armors[:, 3] = armors[:,3] - armors[:,1]
        armors[:, 4] = armors[:,4] - armors[:,2]
        armors = armors[np.logical_not(np.isnan(armors[:,0]))]
        if armors.shape[0] !=0 :
            dph = depth.detect_depth(rects=armors[:,1:].tolist()).reshape(-1,1)
            x0 = (armors[:,1] + armors[:, 3]/2).reshape(-1,1)
            y0 = (armors[:,2] + armors[:, 4]/2).reshape(-1,1)
            xyz = np.concatenate([armors[:,0].reshape(-1,1),x0,y0,dph],axis=1)
            location = np.zeros((10, 4)) * np.nan
            for i in xyz:
                location[int(i[0]),:] = i
            points = alarm.update(location)
            alarm.show()
    if not myshow.view_change:
        pic_info = frame
    else:
        pic_info = frame
    if pic_info is not None:
        myshow.set_image(pic_info, "main_demo")
    if myshow.close:
        if USEABLE['cam_left']:
            Predictor1.stop()
        sys.exit(app.exec_())


if __name__=="__main__":
    from radar_detect.Linar import DepthQueue
    from resources.config import cam_config,test_region,enemy_color,\
        real_size
    from sensor_msgs import point_cloud2
    from sensor_msgs.msg import PointCloud2
    import rosbag
    import cv2
    # ui
    app = QtWidgets.QApplication(sys.argv)
    myshow = Mywindow()
    timer_main = QTimer()
    timer_serial = QTimer()
    myshow.show()

    alarm = Alarm(test_region,lambda x: myshow.set_image(x, "map"),[],enemy_color,real_size,False,True)
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

    for topic, msg, t in bag_data:
        pc = np.float32(point_cloud2.read_points_list(msg, field_names=("x", "y", "z"), skip_nans=True)).reshape(
            -1, 3)
        dist = np.linalg.norm(pc, axis=1)
        pc = pc[dist > 0.4]  # 雷达近距离滤除
        depth.push_back(pc)
        # 显示世界坐标系和相机坐标系坐标和深度，以对测距效果进行粗略测试
    if USEABLE['cam_left']:
        Predictor1 = Predictor('cam_left')
        cam = Camera('cam_left', True)
    timer_main.timeout.connect(spin_once)
    timer_main.start(0)
    sys.exit(app.exec_())
