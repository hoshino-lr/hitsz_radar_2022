"""
配置类
config.py
从外部文件读取配置
同时也存放各类数据
"""
#
color2enemy = {"red": 0, "blue": 1}
enemy2color = ['red', 'blue']

enemy_case = ["diaoshe", 'dafu', 'fei_popre', 'feipo', 'missile_launch1', "missile_launch2"]  # 这些区域在预警时只考虑敌方的该区域
our_case = ["missle_launch1", "missle_lauch2", "danger"]

armor_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'B1', 'B2', 'B3', 'B4', 'B5']  # 雷达站实际考虑的各个装甲板类

unit_list = ['R1', 'R2', 'R3', 'R4', 'R5', 'RG', 'RO', 'RB', 'B1', 'B2', 'B3', 'B4', 'B5', 'BG', 'BO',
             'BB']  # 赛场上各个目标，主要用于HP显示
# 小地图图片路径
MAP_PATH = "resource/map.jpg"
# 小地图设定大小
map_size = (716, 384)
# UI中主视频源初始图像路径
INIT_FRAME_PATH = "resource/demo_pic.jpg"

# 装甲板网络预测的编号和实际预测的编号的对应字典，其中包括用颜色预测出的，对于他们 0 is predicted as blue -1 is predicted as red
_ids = {1: 6, 2: 7, 3: 8, 4: 9, 5: 10, 8: 1, 9: 2, 10: 3, 11: 4, 12: 5, 0: 11, -1: 12}

# 相机序列号
camera_match_list = \
    ['',  # right camera
     '',  # left camera
     '']  # Linar

PC_STORE_DIR = ""
LIDAR_TOPIC_NAME = "/livox/lidar"
