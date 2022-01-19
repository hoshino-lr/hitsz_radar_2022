'''
串口相关操作类
'''
import numpy as np

from official import Game_data_define, official_Judge_Handler
from radar_class.config import enemy,BO

class Port_operate(object):
    _bytes2int = lambda x:(0x0000 | x[0]) | (x[1]<<8)

    _Robot_positions=np.zeros((5,2),dtype=np.float32) #敌方所有机器人坐标
    _Now_stage=0
    _Game_Start_Flag=False
    _Game_End_Flag=False
    Remain_time=0# 剩余时间

    _init_hp=np.ones(10,dtype=int)*500 # 初始血量
    _HP=np.ones(16,dtype=int)*500 # 血量
    _max_hp=_init_hp.copy() # 最大血量 
    _hp_up=np.array([100,150,200,250,300,350,400,450,500]) # 各个升级血量阶段
    _last_hp=_init_hp.copy()
    _set_max_flag=False # 初始化血量

    _BO = 0
    _stage = ["NOT START", "PREPARING", "CHECKING", "5S", "PLAYING", "END"]

    def __init__(self):
        pass

    @staticmethod
    def gain_positions(positions):
        #传入位置
        Port_operate._Robot_positions=np.float32(positions)
    
    @staticmethod
    def positions():
        #传出位置
        return Port_operate._Robot_positions

    @staticmethod
    def Map(targetId,x,y,ser):
        '''
        小地图数据处理
        '''
        buffer=[0]
        buffer*=19

        buffer[0]=0xA5
        buffer[1]=10
        buffer[2]=0
        buffer[3]=1
        buffer[4]=official_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验
        buffer[5]=0x05
        buffer[6]=0x03
        buffer[7]=targetId
        buffer[8]=0
        buffer[9]=bytes(x)[0]
        buffer[10]=bytes(x)[1]
        buffer[11]=bytes(x)[2]
        buffer[12]=bytes(x)[3]
        buffer[13]=bytes(y)[0]
        buffer[14]=bytes(y)[1]
        buffer[15]=bytes(y)[2]
        buffer[16]=bytes(y)[3]
        official_Judge_Handler.Append_CRC16_Check_Sum(id(buffer),19)
        buffer_tmp_array=[0]
        buffer_tmp_array*=19

        for i in range(19):
            buffer_tmp_array[i] = buffer[i]
        ser.write(bytearray(buffer_tmp_array))

    @staticmethod
    def Update_gamedata(buffer):
        if Port_operate._Now_Stage<2 and ((buffer[7]>>4)==2 or (buffer[7]>>4)==3 or (buffer[7]>>4)==4):
            # 从自检阶段开始表示比赛开始
            Port_operate._Game_Start_Flag=True
            Port_operate._set_max_flag=True
        if Port_operate._Now_Stage<5 and (buffer[7]>>4)==5:
            # 比赛结束
            Port_operate._Game_End_Flag=True
            Port_operate._max_hp=Port_operate._init_hp.copy()
        Port_operate._Now_Stage = buffer[7]>>4
        Port_operate.Remain_time = (0x0000 | buffer[8]) | (buffer[9]<<8)
    
    @staticmethod
    def Robot_HP(buffer):
        Port_operate._HP=np.array([Port_operate._bytes2int((buffer[i*2-1],buffer[i*2])) \
            for i in range(4,20)],dtype=int)
        if Port_operate._set_max_flag:
            # 比赛开始时，根据读取血量设置最大血量
            Port_operate._max_hp=Port_operate._HP[[0,1,2,3,4,8,9,10,11,12]]
            Port_operate._set_max_flag=False
        else:
            Port_operate._judge_max_hp()
    
    @staticmethod
    def _judge_max_hp():
        #判断血量
        mask_zero=Port_operate._last_hp>0 # 血量为0不判断
        focus_hp=Port_operate._HP[[0,1,2,3,4,8,9,10,11,12]] # 只关心这些位置的血量上限
        # 工程车血量上限不变，不判断
        mask_engineer=np.array([True]*10)
        mask_engineer[[1,6]]=False
        mask=np.logical_and(mask_zero,mask_engineer)
        # 若血量增加在30到80，则为一级上升（50）
        mask_level1=np.logical_and(focus_hp-Port_operate._last_hp>30,focus_hp-Port_operate._last_hp<=80)
        # 若血量增加在80以上，则为二级上升（100）
        mask_level2=focus_hp-Port_operate._last_hp > 80
        Port_operate._max_hp[np.logical_and(mask_level1,mask)]+=50
        Port_operate._max_hp[np.logical_and(mask_level2,mask)]+=100
        # 如果有一次上限改变没检测到，使得当前血量大于上限，则调整至相应的上限
        mask_still=np.logical_and(focus_hp>Port_operate._max_hp,mask)
        for i in np.where(mask_still)[0]:
            Port_operate._max_hp[i]=np.min(Port_operate._hp_up[Port_operate._hp_up>focus_hp[i]])
        Port_operate._last_hp=focus_hp.copy()

    @staticmethod
    def get_message(hp_scene):
        # 更新hp信息框
        hp_scene.refresh()
        hp_scene.update(Port_operate._HP,Port_operate._max_hp)
        hp_scene.update_stage(Port_operate._stage[Port_operate._Now_Stage],Port_operate.Remain_time,Port_operate._BO+1,BO)

    @staticmethod
    def One_compete_end():
        # 比赛结束
        if Port_operate._Game_End_Flag:
            Port_operate._Game_End_Flag = False
            Port_operate._BO += 1
            return True,Port_operate._BO - BO
        else:
            return False,-1

    @staticmethod
    def One_compete_start():
        # 比赛开始
        if Port_operate._Game_Start_Flag:
            Port_operate._Game_Start_Flag = False
            return True
        else:
            return False

    @staticmethod
    def Receive_Robot_Data(buffer):
        # 车间通信
        if (0x0000 | buffer[7]) | (buffer[8] << 8) == 0x0200:
            print("received")