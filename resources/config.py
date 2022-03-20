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

enemy_color = 0
USEABLE = {
    "cam_left": True,
    "cam_right": True,
    "serial": False,
    "Lidar": False,
}
DEBUG = True
BO = 0
cam_config = {
    "cam_right": {
        "id": "00F78889001",
        "size": (1024, 1024),
        "video_path": "/home/hoshino/CLionProjects/LCR_sjtu/demo_resource/two_cam/1.mp4",
        "K_0": np.mat([[1273.6729986357857, 0.0, 598.3779780737999],
                       [0.0, 1274.0066230685838, 531.2012102435624],
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

net1_onnx = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/net1_sim.onnx"
net1_engine = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/net1.engine"

net2_onnx = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/net2.onnx"
net2_engine = os.path.dirname(os.path.abspath(__file__)) + "/net_onnx/net2.engine"
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

test_objNames = ['R4下来的角', '坡除角', '坡上角', '烧饼轨道 u远',
                 '烧饼轨道 近', '坡，另一边角']
test_objPoints = np.array([[4021, 3994, 0],  # R4下来的角
                           [7820, 4550, 0],  # 坡除角
                           [6756, 6300, 600],  # 坡上角
                           [2250, 8311, 1376],  # 烧饼轨道 u远
                           [2250, 4825, 1376],  # 烧饼轨道 近
                           [6970, 8311, 600]], dtype=np.float32)  # 坡，另一边角


class LOGGER():
    """
    logger 类
    """

    def __init__(self, text_api):
        # 创建一个logger
        import time
        logger_name = time.strftime('%Y-%m-%d %H-%M-%S')
        self.logger = logging.getLogger(logger_name)
        self.logger.setLevel(logging.DEBUG)
        self.api = text_api
        # 创建一个handler，用于写入日志文件
        log_path = os.path.abspath(os.getcwd()) + "/logs/"  # 指定文件输出路径
        if not os.path.exists(log_path):
            os.makedirs(log_path)
        logname = log_path + logger_name + '.log'  # 指定输出的日志文件名
        fh = logging.FileHandler(logname, encoding='utf-8')  # 指定utf-8格式编码，避免输出的日志文本乱码
        fh.setLevel(logging.DEBUG)

        # 创建一个handler，用于将日志输出到控制台
        ch = logging.StreamHandler()
        ch.setLevel(logging.DEBUG)

        # 定义handler的输出格式
        formatter = logging.Formatter('%(asctime)s-%(levelname)s-%(message)s')
        fh.setFormatter(formatter)
        ch.setFormatter(formatter)

        # 给logger添加handler
        self.logger.addHandler(fh)
        self.logger.addHandler(ch)

    def add_text(self, handle, text, show_api=True):
        """
        :param handle 警告类型 有 WARNING ERROR INFO 三种
        :param text 输入的文字
        :param show_api 是否显示在 ui 上
        """
        if handle == "WARNING":
            self.logger.warning(text)
        if handle == "ERROR":
            self.logger.error(text)
        if handle == "INFO":
            self.logger.info(text)
        if show_api:
            self.api(handle, text)
