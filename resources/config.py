"""
配置类
config.py
从外部文件读取配置
created by 李龙 2021/12
最终修改 by 李龙 2021/1/15
"""
import os
import numpy as np
import logging

logger = logging.Logger("test")
absolute_path = os.path.dirname(os.path.abspath(__file__))
enemy_color = 0
USEABLE = {
    "cam_left": True,
    "cam_right": False,
    "serial": False,
    "Lidar": False,
}
DEBUG = True

cam_config = {
    "cam_right": {
        "id": "00F78889001",
        "size": (1024, 1024),
        "video_path": "/home/hoshino/CLionProjects/hitsz_radar/resources/two_cam/mangdao.avi",
        "K_0": np.mat([
            [1269.249100458914, 0.0, 595.4045131856898],
            [0, 1269.5641866741526, 534.0522814630621],
            [0, 0, 1],
        ]),
        "C_0": np.mat([-0.2275028300247, 0.20188387553073965, -0.00032941427232237167, -0.0007610612612672920245, 0.09717811036906197]),
        "rvec": np.mat([[1.59668528], [0.58626031], [-0.53932911]]),
        "tvec": np.mat([[-8625.00028137], [771.3457855], [6926.60950051]]),
        "E_0": np.mat([
            [0.0474247,-0.998873,-0.0019402, -0.00701503],
            [0.12093, 0.00766964, -0.992631, -0.0844397],
            [0.991528, 0.0468406, 0.121157, 0.0749353,],
            [0,0,0,1]
        ])},

    "cam_left": {
        "id": "00F78889001",
        "size": (1024, 1024),
        "video_path": "/home/hoshino/CLionProjects/hitsz_radar/resources/two_cam/mangdao.avi",
        "K_0": np.mat([[1273.6729986357857, 0.0, 598.3779780737999],
                       [0.0, 1274.0066230685838, 531.2012102435624],
                       [0.0, 0.0, 1.0]] ),
        "C_0": np.mat([[-0.22753846151806761], [0.2209031621277345], [-0.0006069352871209068], [-0.0006361387371312384], [0.02412961227405689]]),
        "rvec": np.mat([[ 1.59668528], [0.58626031], [-0.53932911]]),
        "tvec": np.mat([[-8625.00028137], [771.3457855], [6926.60950051]]),
        "E_0": np.mat([
            [0.0474247,-0.998873,-0.0019402, -0.00701503],
            [0.12093, 0.00766964, -0.992631, -0.0844397],
            [0.991528, 0.0468406, 0.121157, 0.0749353],
            [0,0,0,1]
        ])}
}

net1_onnx = "net1_sim.onnx"
net1_engine = "net1.engine"

net2_onnx = "net2.onnx"
net2_engine = "net2.engine"
net1_cls = ['car', 'watcher', 'base']

net2_cls_names = ["0", "1", "2", "3", "4",
                  "5", "O", "Bs", "Bb"]

net2_col_names = ["B", "R", "N", "P"]

color2enemy = {"red": 0, "blue": 1}

enemy2color = ['blue', 'red']

enemy_case = ["gaodi","mangdao"]  # 这些区域在预警时只考虑敌方的该区域

our_case = ["missle_launch1", "missle_lauch2", "danger"]

armor_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'B1', 'B2', 'B3', 'B4', 'B5']  # 雷达站实际考虑的各个装甲板类

unit_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'RG', 'RO', 'RB', 'B1', 'B2', 'B3', 'B4', 'B5', 'BG', 'BO',
             'BB']  # 赛场上各个目标，主要用于HP显示

# 小地图图片路径
MAP_PATH = "/home/hoshino/CLionProjects/hitsz_radar/resources/map.jpg"
BAG_FIRE = "/home/hoshino/CLionProjects/camera_lidar_calibration/data/game/beijing.bag"

# 小地图设定大小
map_size = (716, 384)
real_size = (28.,15.)
# UI中主视频源初始图像路径

INIT_FRAME_PATH = "demo_pic.jpg"

# 裁判系统发送编号定义
loc2code = \
    {
        'dart': 0,
        'feipo': 1,
        'feipopre': 2,
        "gaodipre": 3,  # xinjiapo
        "gaodipre2": 4,  # goubipo
        'base': 5,

    }
loc2car = {
    'dart': [7],
    'feipopre': [1],
    'feipo': [1, 3, 4, 5],
    'gaodipre': [1, 3, 4, 5],
    'gaodipre2': [1, 3, 4, 5],
    'base': [1, 2, 3, 4, 5]
}

region = \
    {

        'm_l_red_base_r': [5.500, 9.700, 5.500, 5.300, 0.500],  # red base only for enemy 1
        'm_l_blue_base_l': [22.500, 9.700, 22.500, 5.300, 0.500],  # blue base only for enemy 0
        'a_fp_red_diaoshe_a': [8.682, 13.906, 6.101, 10.117, 4.536, 10.156, 4.575, 14.023, 0.45, 0.],
        # enemy case,only alarm
        'a_fp_blue_diaoshe_a': [23.464, 4.844, 23.425, 0.977, 19.318, 1.094, 21.899, 4.883, 0.45, 0.],
        'm_r_red_dafu_a': [7.718, 2.532, 9.130, 1.315, 0.945, 0.],  # enemy case, only alarm
        'm_r_red_feipopre_a': [2.855, 3.516, 4.184, 0.000, 0.4, 0.],  # only enemy case, only alarm
        'a_r_red_feipo_a': [4.536, 1.016, 12.318, 0.156, 0.3, 0.],
        # only enemy case, send and alarm : send to 1,3,4,5 TASK 2
        'm_r_blue_dafu_a': [18.870, 13.685, 20.282, 12.468, 0.945, 0.],
        'm_r_blue_feipopre_a': [23.816, 15.000, 25.145, 11.484, 0.4, 0.],
        'a_r_blue_feipo_a': [15.682, 14.844, 23.464, 13.984, 0.3, 0.],
        'm_fp_red_gaodipre_d': [10.421, 7.183, 11.614, 5.357, 10.421, 4.554, 8.960, 6.721, 0.6, 0.],
        # every case, send and alarm : send to 1,3,4,5 TASK 2
        'm_fp_blue_gaodipre_d': [17.579, 10.446, 19.040, 8.279, 17.579, 7.817, 16.386, 9.643, 0., 0.6],  # xinjiapo
        'm_fp_red_gaodipre2_a': [13.531, 13.906, 11.497, 11.211, 10.559, 11.836, 12.318, 14.492, 0., 0.],
        # every case, send and alarm : send to 1,3,4,5 TASK 2
        'm_fp_blue_gaodipre2_a': [17.441, 3.164, 15.682, 0.508, 14.469, 1.094, 16.503, 3.789, 0., 0.],  # goubipo
        's_fp_red_missilelaunch2_d': [0.365, 11.761, 1.266, 11.810, 1.266, 9.789, 0.243, 9.862, 4, 2.4],
        's_fp_red_missilelaunch1_d': [1.290, 11.323, 0.609, 11.347, 0.682, 10.300, 1.290, 10.252, 1.1, 0.4],
        's_fp_blue_missilelaunch2_d': [26.734, 5.211, 27.757, 5.138, 27.635, 3.239, 26.734, 3.190, 2.4, 4],
        's_fp_blue_missilelaunch1_d': [27.318, 4.700, 26.710, 4.748, 26.710, 3.677, 27.391, 3.653, 0.4, 1.1]
    }
# 经过转换过的区域定义 (28.,15.) -> (12.,6.) 转换函数见 tools/generate_region.py
test_region = \
    {
        's_fp_red_gaodi_a': [6.756,6.300,5.871,5.764,5.871,8.471,6.756,8.311,0.600],
        's_fp_blue_gaodi_a': [6.756,6.300,5.871,5.764,5.871,8.471,6.756,8.311,0.600],
        's_fp_red_mangdao_a': [5.854,2.630,3.320,4.718,5.871,5.871,6.756,4.021,0],
        's_fp_blue_mangdao_a': [5.854,2.630,3.320,4.718,5.871,5.871,6.756,4.021,0]
    }

test_scene = \
    {}
PC_STORE_DIR = ""
LIDAR_TOPIC_NAME = "/livox/lidar"

def config_init():
    global net1_onnx
    global net2_onnx
    global net1_engine
    global net2_engine
    global INIT_FRAME_PATH
    global MAP_PATH
    net1_onnx = os.path.join(absolute_path, net1_onnx)
    net2_onnx = os.path.join(absolute_path, net2_onnx)
    net1_engine = os.path.join(absolute_path, net1_engine)
    net2_engine = os.path.join(absolute_path, net2_engine)
    INIT_FRAME_PATH = os.path.join(absolute_path, INIT_FRAME_PATH)
    MAP_PATH = os.path.join(absolute_path, MAP_PATH)
