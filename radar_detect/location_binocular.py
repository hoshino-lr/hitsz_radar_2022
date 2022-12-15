import cv2 as cv
import numpy as np
from config import cam_config
from mapping.drawing import draw_message
import time
import pickle as pkl
from radar_detect.common import is_inside_polygon
import mapping.draw_map as draw_map  # 引入draw_map模块，使用其中的CompeteMap类
from config import armor_list, color2enemy, enemy_color, cam_config, real_size, region, test_region, choose
from radar_detect.location_Delaunay import location_Delaunay

class location_binocular:
    init_ok = False
    w_points = None
    cam_rect = ()
    rect_points = None
    debug_mode = False