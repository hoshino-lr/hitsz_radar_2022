"""
配置类
config.py
从外部文件读取配置
created by 李龙 2021/12
最终修改 by 李龙 2021/1/15
"""
import os
import numpy as np

enemy_color = 0
USEABLE = {
    "cam_left": True,
    "cam_right": False,
    "cam_far": False,
    "serial": False,
    "Lidar": False,
    "using_d": False,
}
DEBUG = True
BO = 0
cam_config = {
    "cam_right": {
        "id": "00F78889001",
        "size": (3072, 2048),
        "video_path": "/home/mark/demo_resource/two_cam/1.mp4",
        "K_0": np.mat([[2505.2629026927225, 0.0, 1529.4286325395244],
                       [0.0, 2505.5722700649067, 1026.1378217662113],
                       [0.0, 0.0, 1.0]]),
        "C_0": np.mat([-0.2275028300247, 0.20188387553073965, -0.00032941427232237167, -0.0007610612612672920245,
                       0.09717811036906197]),
        "exposure": 5000,
        "gain": 15,
        "rvec": np.mat([[1.59668528], [0.58626031], [-0.53932911]]),
        "tvec": np.mat([[-8625.00028137], [771.3457855], [6926.60950051]]),
        "E_0": np.mat([
            [0.0474247, -0.998873, -0.0019402, -0.00701503],
            [0.12093, 0.00766964, -0.992631, -0.0844397],
            [0.991528, 0.0468406, 0.121157, 0.0749353, ],
            [0, 0, 0, 1]
        ])},

    "cam_left": {
        "id": "J37877236",
        "size": (3072, 2048),
        "video_path": "/home/mark/demo_resource/two_cam/2.mp4",
        "K_0": np.mat([[2551.020329744114, 0.0, 1495.9469657967131],
                       [0.0, 2552.472255441562, 1062.7407379282765],
                       [0.0, 0.0, 1.0]]),
        "C_0": np.mat([[-0.22753846151806761], [0.2209031621277345], [-0.0006069352871209068], [-0.0006361387371312384],
                       [0.02412961227405689]]),
        "exposure": 5000,
        "gain": 15,
        "rvec": np.mat([[1.59668528], [0.58626031], [-0.53932911]]),
        "tvec": np.mat([[-8625.00028137], [771.3457855], [6926.60950051]]),
        "E_0": np.mat([
            [0.0474247, -0.998873, -0.0019402, -0.00701503],
            [0.12093, 0.00766964, -0.992631, -0.0844397],
            [0.991528, 0.0468406, 0.121157, 0.0749353],
            [0, 0, 0, 1]
        ])},

    "cam_far": {
        "id": "00F78889001",
        "size": (1024, 1024),
        "video_path": "/home/hoshino/CLionProjects/LCR_sjtu/demo_resource/two_cam/2.mp4",
        "K_0": np.mat([[1273.6729986357857, 0.0, 598.3779780737999],
                       [0.0, 1274.0066230685838, 531.2012102435624],
                       [0.0, 0.0, 1.0]]),
        "C_0": np.mat([[-0.22753846151806761], [0.2209031621277345], [-0.0006069352871209068], [-0.0006361387371312384],
                       [0.02412961227405689]]),
        "exposure": 5000,
        "gain": 15,
        "rvec": np.mat([[1.59668528], [0.58626031], [-0.53932911]]),
        "tvec": np.mat([[-8625.00028137], [771.3457855], [6926.60950051]]),
        "E_0": np.mat([
            [0.0474247, -0.998873, -0.0019402, -0.00701503],
            [0.12093, 0.00766964, -0.992631, -0.0844397],
            [0.991528, 0.0468406, 0.121157, 0.0749353],
            [0, 0, 0, 1]
        ])}

}

net1_onnx = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/best.onnx"
net1_engine = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/net1.engine"

net2_onnx = os.path.dirname(os.path.abspath(__file__)) + "/models/net2.onnx"
net2_engine = os.path.dirname(os.path.abspath(__file__)) + "/models/net2.engine"
net1_cls = ['car', 'watcher', 'base']

net2_cls_names = ["0", "1", "2", "3", "4",
                  "5", "O", "Bs", "Bb"]

net2_col_names = ["B", "R", "N", "P"]

color2enemy = {"red": 0, "blue": 1}

enemy2color = ['blue', 'red']

enemy_case = ["gaodi", "mangdao"]  # 这些区域在预警时只考虑敌方的该区域

our_case = ["missle_launch1", "missle_lauch2", "danger"]

armor_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'B1', 'B2', 'B3', 'B4', 'B5']  # 雷达站实际考虑的各个装甲板类

unit_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'RG', 'RO', 'RB', 'B1', 'B2', 'B3', 'B4', 'B5', 'BG', 'BO',
             'BB']  # 赛场上各个目标，主要用于HP显示

# 小地图图片路径
MAP_PATH = os.path.dirname(os.path.abspath(__file__)) + "/map.jpg"
BAG_FIRE = "/home/hoshino/CLionProjects/camera_lidar_calibration/data/game/beijing.bag"

# 小地图设定大小
map_size = (716, 384)
real_size = (28., 15.)
# UI中主视频源初始图像路径

INIT_FRAME_PATH = os.path.dirname(os.path.abspath(__file__)) + "/beijing.png"

region = \
    {}
# 经过转换过的区域定义 (28.,15.) -> (12.,6.) 转换函数见 tools/generate_region.py
test_region = \
    {
        's_fp_red_gaodi_a': [6756, 6300, 5871, 5764, 5871, 8471, 6756, 8311, 600],
        's_fp_blue_gaodi_a': [6756, 6300, 5871, 5764, 5871, 8471, 6756, 8311, 600],
        's_fp_red_mangdao_a': [5854, 2630, 3320, 4718, 5871, 5871, 6756, 4021, 0],
        's_fp_blue_mangdao_a': [5854, 2630, 3320, 4718, 5871, 5871, 6756, 4021, 0]
    }

PC_STORE_DIR = ""
LIDAR_TOPIC_NAME = "/livox/lidar"

# 0为正式场上使用的points 1为debug用
objNames = [
    {
        "left": ['R4下来的角', '坡除角', '坡上角', '烧饼轨道 u远',
                 '烧饼轨道 近', '坡，另一边角'],
        "right": ['R4下来的角', '坡除角', '坡上角', '烧饼轨道 u远',
                  '烧饼轨道 近', '坡，另一边角']
    },
    {
        "left": ['R4下来的角', '坡除角', '坡上角', '烧饼轨道 u远',
                 '烧饼轨道 近', '坡，另一边角'],
        "right": ['R4下来的角', '坡除角', '坡上角', '烧饼轨道 u远',
                  '烧饼轨道 近', '坡，另一边角']
    }
]

# 0为正式场上使用的points 1为debug用
objPoints = [
    {
        "cam_left": np.array([[4021, 3994, 0],  # R4下来的角
                              [7820, 4550, 0],  # 坡除角
                              [6756, 6300, 600],  # 坡上角
                              [2250, 8311, 1376],  # 烧饼轨道 u远
                              [2250, 4825, 1376],  # 烧饼轨道 近
                              [6970, 8311, 600]], dtype=np.float32),  # 坡，另一边角
        "cam_right": np.array([[4021, 3994, 0],  # R4下来的角
                               [7820, 4550, 0],  # 坡除角
                               [6756, 6300, 600],  # 坡上角
                               [2250, 8311, 1376],  # 烧饼轨道 u远
                               [2250, 4825, 1376],  # 烧饼轨道 近
                               [6970, 8311, 600]], dtype=np.float32),  # 坡，另一边角
    },
    {
        "cam_left": np.array([[4021, 3994, 0],  # R4下来的角
                              [7820, 4550, 0],  # 坡除角
                              [6756, 6300, 600],  # 坡上角
                              [2250, 8311, 1376],  # 烧饼轨道 u远
                              [2250, 4825, 1376],  # 烧饼轨道 近
                              [6970, 8311, 600]], dtype=np.float32),  # 坡，另一边角
        "cam_right": np.array([[4021, 3994, 0],  # R4下来的角
                               [7820, 4550, 0],  # 坡除角
                               [6756, 6300, 600],  # 坡上角
                               [2250, 8311, 1376],  # 烧饼轨道 u远
                               [2250, 4825, 1376],  # 烧饼轨道 近
                               [6970, 8311, 600]], dtype=np.float32),  # 坡，另一边角
    }
]

# 0为正式场上使用的points 1为debug用
Delaunary_points = [
    [
        {
            "cam_left": [
                (),
                [
                    # (world_points)
                ]

            ],
            "cam_right": {
                (),
                [
                    # (world_points)
                ]
            }
        }
    ],
    [
        {
            "cam_left": [
                (),
                [
                    # (world_points)
                ]

            ],
            "cam_right": {
                (),
                [
                    # (world_points)
                ]
            }
        }
    ]
]
