'''
雷达监测类
back_projection.py
用于在图像中寻找对方车车。
'''

class b_projection(object):
    """
    反投影类
    """
    # 类全局变量
    _using_bbox = True  # 若为真，则使用装甲板四点的bounding box四点作为落入预警区域的判断依据
    _twinkle_times = 3  # 闪烁几次
    _iou_cache = False  # 存不存用IOU预测出来的装甲板（详见技术报告）
    _iou_thre = 0.8  # 只有高于IoU阈值的才会被预测

    def __init__(self,src,region,enemy,debug=False):
        """
        :param src:输入图像
        """
        self.enery_color = enemy
        self.src = src
    
    def update(self):
        pass
