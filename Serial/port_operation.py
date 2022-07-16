"""
串口相关操作类
"""
import numpy as np
import time
from .official import Game_data_define, official_Judge_Handler
from config import enemy_color, BO


class Port_operate(object):
    _bytes2int = lambda x: (0x0000 | x[0]) | (x[1] << 8)

    _Robot_positions = np.zeros((5, 2), dtype=np.float32)  # 敌方所有机器人坐标
    _Robot_positions_us = np.zeros((5, 2), dtype=np.float32)  # 敌方所有机器人坐标
    _Robot_decisions = np.zeros((5, 2), dtype=np.float32)  # 我方所有机器人状态
    _Now_stage = 0
    _Now_state = 0
    _last_state = 0
    _Game_Start_Flag = False
    _Game_End_Flag = False
    Remain_time = 0  # 剩余时间

    change_view = -1
    highlight = False
    missile_bit = False

    _init_hp = np.ones(10, dtype=int) * 500  # 初始血量
    _HP = np.ones(16, dtype=int) * 500  # 血量
    _max_hp = _init_hp.copy()  # 最大血量
    _hp_up = np.array([100, 150, 200, 250, 300, 350, 400, 450, 500])  # 各个升级血量阶段
    _last_hp = _init_hp.copy()
    _set_max_flag = False  # 初始化血量

    _BO = 0
    _stage = ["NOT START", "PREPARING", "CHECKING", "5S", "PLAYING", "END"]

    _state = ["normal", "small_energy", "big_energy"]

    _energy_time = 0

    def __init__(self):
        pass

    @staticmethod
    def gain_positions(positions):
        # 传入位置
        Port_operate._Robot_positions = np.float32(positions)

    @staticmethod
    def gain_decisions(decisions):
        # 传入位置
        Port_operate._Robot_decisions = np.float32(decisions)

    @staticmethod
    def positions():
        # 传出位置
        return Port_operate._Robot_positions

    @staticmethod
    def positions_us():
        # 传出位置
        return Port_operate._Robot_positions_us

    @staticmethod
    def decisions():
        # 传出位置
        return Port_operate._Robot_decisions

    @staticmethod
    def get_state():
        # 传出位置
        return [Port_operate._Now_state, Port_operate._energy_time]

    @staticmethod
    def HP():
        return Port_operate._HP

    @staticmethod
    def Map(targetId, x, y, ser):
        """
        小地图数据处理
        """
        buffer = [0]
        buffer *= 19
        Port_operate.generate_head(buffer, length=10)
        buffer[5] = 0x05
        buffer[6] = 0x03
        buffer[7] = targetId
        buffer[8] = 0
        buffer[9] = bytes(x)[0]
        buffer[10] = bytes(x)[1]
        buffer[11] = bytes(x)[2]
        buffer[12] = bytes(x)[3]
        buffer[13] = bytes(y)[0]
        buffer[14] = bytes(y)[1]
        buffer[15] = bytes(y)[2]
        buffer[16] = bytes(y)[3]
        official_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 19)
        buffer_tmp_array = [0]
        buffer_tmp_array *= 19
        for i in range(19):
            buffer_tmp_array[i] = buffer[i]
        ser.write(bytearray(buffer_tmp_array))

    @staticmethod
    def Update_gamedata(buffer):
        if Port_operate._Now_stage < 2 and ((buffer[7] >> 4) == 2 or (buffer[7] >> 4) == 3 or (buffer[7] >> 4) == 4):
            # 从自检阶段开始表示比赛开始
            Port_operate._Game_Start_Flag = True
            Port_operate._set_max_flag = True
        if Port_operate._Now_stage < 5 and (buffer[7] >> 4) == 5:
            # 比赛结束
            Port_operate._Game_End_Flag = True
            Port_operate._max_hp = Port_operate._init_hp.copy()
        Port_operate._Now_stage = buffer[7] >> 4
        Port_operate.Remain_time = (0x0000 | buffer[8]) | (buffer[9] << 8)

    @staticmethod
    def Robot_HP(buffer):
        Port_operate._HP = np.array([Port_operate._bytes2int((buffer[i * 2 - 1], buffer[i * 2])) \
                                     for i in range(4, 20)], dtype=int)
        if Port_operate._set_max_flag:
            # 比赛开始时，根据读取血量设置最大血量
            Port_operate._max_hp = Port_operate._HP[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12]]
            Port_operate._set_max_flag = False
        else:
            Port_operate._judge_max_hp()

    @staticmethod
    def _judge_max_hp():
        # 判断血量
        mask_zero = Port_operate._last_hp > 0  # 血量为0不判断
        focus_hp = Port_operate._HP[[0, 1, 2, 3, 4, 8, 9, 10, 11, 12]]  # 只关心这些位置的血量上限
        # 工程车血量上限不变，不判断
        mask_engineer = np.array([True] * 10)
        mask_engineer[[1, 6]] = False
        mask = np.logical_and(mask_zero, mask_engineer)
        # 若血量增加在30到80，则为一级上升（50）
        mask_level1 = np.logical_and(focus_hp - Port_operate._last_hp > 30, focus_hp - Port_operate._last_hp <= 80)
        # 若血量增加在80以上，则为二级上升（100）
        mask_level2 = focus_hp - Port_operate._last_hp > 80
        Port_operate._max_hp[np.logical_and(mask_level1, mask)] += 50
        Port_operate._max_hp[np.logical_and(mask_level2, mask)] += 100
        # 如果有一次上限改变没检测到，使得当前血量大于上限，则调整至相应的上限
        mask_still = np.logical_and(focus_hp > Port_operate._max_hp, mask)
        for i in np.where(mask_still)[0]:
            Port_operate._max_hp[i] = np.min(Port_operate._hp_up[Port_operate._hp_up > focus_hp[i]])
        Port_operate._last_hp = focus_hp.copy()

    @staticmethod
    def get_message(hp_scene):
        # 更新hp信息框
        hp_scene.refresh()
        hp_scene.update(Port_operate._HP, Port_operate._max_hp)
        hp_scene.update_stage(Port_operate._stage[Port_operate._Now_stage], Port_operate.Remain_time,
                              Port_operate._BO + 1, BO)

    @staticmethod
    def One_compete_end():
        # 比赛结束
        if Port_operate._Game_End_Flag:
            Port_operate._Game_End_Flag = False
            Port_operate._BO += 1
            return True, Port_operate._BO - BO
        else:
            return False, -1

    @staticmethod
    def One_compete_start():
        # 比赛开始
        if Port_operate._Game_Start_Flag:
            Port_operate._Game_Start_Flag = False
            return True
        else:
            return False

    @staticmethod
    def Update_robo_position(buffer):
        if Port_operate._Now_stage < 2 and ((buffer[7] >> 4) == 2 or (buffer[7] >> 4) == 3 or (buffer[7] >> 4) == 4):
            # 从自检阶段开始表示比赛开始
            Port_operate._Game_Start_Flag = True
            Port_operate._set_max_flag = True
        if Port_operate._Now_stage < 5 and (buffer[7] >> 4) == 5:
            # 比赛结束
            Port_operate._Game_End_Flag = True
            Port_operate._max_hp = Port_operate._init_hp.copy()
        Port_operate._Now_stage = buffer[7] >> 4
        Port_operate.Remain_time = (0x0000 | buffer[8]) | (buffer[9] << 8)

    @staticmethod
    def Receive_Robot_Data(buffer):
        # 车间通信
        if Port_operate._Game_Start_Flag:
            sender_id = (buffer[8] << 8) | buffer[7]
            if sender_id == 0x106 or sender_id == 0x6:
                Port_operate.change_view = buffer[13]
                Port_operate.highlight = buffer[14]
                Port_operate.missile_bit = buffer[15]
                if Port_operate.missile_bit:
                    Port_operate._energy_time = time.time()
                    if Port_operate.Remain_time < 240:
                        Port_operate._Now_state = 4
                    else:
                        Port_operate._Now_state = 3

            elif sender_id >= 0x100:
                Port_operate._Robot_positions_us[sender_id - 0x100] = np.array([buffer[13], buffer[14]])
            else:
                Port_operate._Robot_positions_us[sender_id] = np.array([buffer[13], buffer[14]])
        else:
            Port_operate.change_view = -1
            Port_operate.Robot_positions_us = np.zeros((5, 2), dtype=np.float32)  # 敌方所有机器人坐标

    @staticmethod
    def Receive_State_Data(buffer):
        # 能量机关状态
        s_energy = (buffer[7] >> 4) & 0x1
        b_energy = (buffer[7] >> 5) & 0x1
        if s_energy == 1:
            Port_operate._Now_state = 1
        elif b_energy == 1:
            Port_operate._Now_state = 2
        else:
            Port_operate._Now_state = 0
        if Port_operate._last_state != Port_operate._Now_state:
            if Port_operate._Now_state == 1:
                Port_operate._energy_time = time.time()
            elif Port_operate._Now_state == 1:
                Port_operate._energy_time = time.time()
            else:
                Port_operate._energy_time = time.time()
            Port_operate._last_state = Port_operate._Now_state
        else:
            if time.time() - Port_operate._energy_time >= 45:
                Port_operate._Now_state = -1
                Port_operate._energy_time = time.time()
            if Port_operate._Now_state == -1 and time.time() - Port_operate._energy_time >= 30:
                Port_operate._Now_state = 0
            Port_operate._last_state = Port_operate._Now_state

    @staticmethod
    def generate_head(buffer, length):
        buffer[0] = 0xA5
        buffer[1] = length
        buffer[2] = 0
        buffer[3] = 1
        buffer[4] = official_Judge_Handler.myGet_CRC8_Check_Sum(id(buffer), 5 - 1, 0xff)  # 帧头 CRC8 校验

    @staticmethod
    def robo_alarm(target_id, my_id, alarm_type: int, attack_target: int, ser):
        # 车间通信
        buffer = [0]
        buffer *= 19
        Port_operate.generate_head(buffer, length=10)
        buffer[5] = 0x01
        buffer[6] = 0x03
        buffer[7] = 0x10
        buffer[8] = 0x02
        buffer[9] = my_id
        buffer[10] = 0
        buffer[11] = target_id
        buffer[12] = 0
        buffer[13] = alarm_type  # 预警信息 0：normal 1：attack 2：run 3：fly
        buffer[14] = attack_target
        buffer[15] = 0  # 上下云台
        buffer[16] = 0  # 方位
        official_Judge_Handler.Append_CRC16_Check_Sum(id(buffer), 19)
        buffer_tmp_array = [0]
        buffer_tmp_array *= 19
        for i in range(19):
            buffer_tmp_array[i] = buffer[i]
        ser.write(bytearray(buffer_tmp_array))

    @staticmethod
    def Map_Transmit(ser):
        # 画小地图
        position = Port_operate.positions()[Port_operate.Map_Transmit.nID]
        x, y = position
        # 坐标为零则不发送
        if np.isclose(position, 0).all():
            flag = False
        else:
            flag = True
        # 敌方判断
        if enemy_color == 0:
            # 敌方为红方
            if flag:
                Port_operate.Map(Port_operate.Map_Transmit.r_id, np.float32(x), np.float32(y), ser)
                time.sleep(0.1)
                if Port_operate.Map_Transmit.r_id == 5:
                    Port_operate.Map_Transmit.r_id = 1
                else:
                    Port_operate.Map_Transmit.r_id += 1
        if enemy_color == 1:
            # 敌方为蓝方
            if flag:
                Port_operate.Map(Port_operate.Map_Transmit.b_id, np.float32(x), np.float32(y), ser)
                time.sleep(0.1)
                if Port_operate.Map_Transmit.b_id == 105:
                    Port_operate.Map_Transmit.b_id = 101
                else:
                    Port_operate.Map_Transmit.b_id += 1
        Port_operate.Map_Transmit.nID = (Port_operate.Map_Transmit.nID + 1) % 5

    @staticmethod
    def port_send(ser):
        Port_operate.Map_Transmit(ser)
        Port_operate.port_manager(ser)

    @staticmethod
    def port_send_init():
        if not hasattr(Port_operate.port_manager, 'r_id'):
            Port_operate.port_manager.r_id = 1
        if not hasattr(Port_operate.port_manager, 'b_id'):
            Port_operate.port_manager.b_id = 101
        if not hasattr(Port_operate.port_manager, 'nID'):
            Port_operate.port_manager.nID = 0
        if not hasattr(Port_operate.Map_Transmit, 'r_id'):
            Port_operate.Map_Transmit.r_id = 1
        if not hasattr(Port_operate.Map_Transmit, 'b_id'):
            Port_operate.Map_Transmit.b_id = 101
        if not hasattr(Port_operate.Map_Transmit, 'nID'):
            Port_operate.Map_Transmit.nID = 0

    @staticmethod
    def port_manager(ser):

        # 英雄飞坡预警
        alarm_type = int(Port_operate.decisions()[Port_operate.port_manager.nID][0])
        attack_target = int(Port_operate.decisions()[Port_operate.port_manager.nID][1])
        # 敌方判断
        if enemy_color == 0:
            # 敌方为红方
            my_id = 109
            Port_operate.robo_alarm(Port_operate.port_manager.b_id, my_id, alarm_type, attack_target, ser)
            time.sleep(0.1)
            if Port_operate.port_manager.b_id == 105:
                Port_operate.port_manager.b_id = 101
            else:
                Port_operate.port_manager.b_id += 1
        if enemy_color == 1:
            # 敌方为蓝方
            my_id = 9
            Port_operate.robo_alarm(Port_operate.port_manager.b_id, my_id, alarm_type, attack_target, ser)
            time.sleep(0.1)
            if Port_operate.port_manager.r_id == 5:
                Port_operate.port_manager.r_id = 1
            else:
                Port_operate.port_manager.r_id += 1
        Port_operate.port_manager.nID = (Port_operate.port_manager.nID + 1) % 5
