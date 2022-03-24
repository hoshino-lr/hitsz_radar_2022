"""
串口通信
"""
import time

import numpy as np
from .official import official_Judge_Handler, Game_data_define
from .port_operation import Port_operate
from resources.config import enemy_color
from .HP_show import HP_scene

buffer = [0]
bufferbyte = 0
cmd_id = 0
r_id = 1
b_id = 101
nID = 0

Game_state = Game_data_define.game_state()
Game_result = Game_data_define.game_result()
Game_robot_HP = Game_data_define.game_robot_HP()
Game_dart_status = Game_data_define.dart_status()
Game_event_data = Game_data_define.event_data()
Game_supply_projectile_action = Game_data_define.supply_projectile_action()
Game_refree_warning = Game_data_define.refree_warning()
Game_dart_remaining_time = Game_data_define.dart_remaining_time()


def read_init(b):
    global bufferbyte
    bufferbyte = 0
    if b[bufferbyte] == 0xa5:
        bufferbyte = 1


def read(ser):
    global buffer
    global bufferbyte
    global cmd_id
    cmd_use = {0x0001, 0x0002, 0x0003, 0x0101, 0x0105}
    bufferbyte = 0
    while True:
        s = int().from_bytes(ser.read(1), 'big')

        if bufferbyte > 50:
            bufferbyte = 0
        buffer[bufferbyte] = s

        # 对帧头进行处理
        if bufferbyte == 0:
            if buffer[bufferbyte] != 0xa5:
                bufferbyte = 0
                continue

        if bufferbyte == 4:
            if official_Judge_Handler.myVerify_CRC8_Check_Sum(id(buffer), 5) == 0:
                read_init(buffer)
                continue

        # 处理cmd_id
        if bufferbyte == 6:
            cmd_id = (0x0000 | buffer[5]) | (buffer[6] << 8)

        # 对内容进行判断
        if not cmd_id in cmd_use:
            bufferbyte = 0
            continue
            # 比赛状态数据
        if bufferbyte == 19 and cmd_id == 0x0001:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 20):
                # 处理比赛信息
                Port_operate.Update_gamedata(buffer)
                read_init(buffer)
                continue

        # 比赛结果数据
        if bufferbyte == 9 and cmd_id == 0x0002:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                # 处理比赛结果
                Game_result.winner = buffer[7]
                read_init(buffer)
                continue

        # 机器人血量统计
        if bufferbyte == 40 and cmd_id == 0x0003:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 40):
                # 统计血量
                Port_operate.Robot_HP(buffer)
                read_init(buffer)
                continue

        # 场地事件数据
        if bufferbyte == 12 and cmd_id == 0x0101:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Game_event_data.event_type = [buffer[7], buffer[8], buffer[9], buffer[10]]  # 储存但未使用
                read_init(buffer)
                continue

        # 飞镖发射倒计时
        if bufferbyte == 9 and cmd_id == 0x0105:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Game_dart_remaining_time.time = buffer[7]
                read_init(buffer)
                continue

        # 云台手通信
        if bufferbyte == 24 and cmd_id == 0x0303:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 24):
                # 此处加入处理
                read_init(buffer)
                continue

        bufferbyte += 1


def Map_Transmit(ser):
    # 画小地图
    global r_id, b_id, nID
    loop_send = 0
    position = Port_operate.positions()[nID]
    x, y = position
    # 坐标为零则不发送
    if np.isclose(position, 0).all():
        flag = False
    else:
        flag = True
    # 敌方判断
    if enemy_color == 1:
        # 敌方为红方
        if flag:
            Port_operate.Map(r_id, np.float32(x), np.float(y), ser)
            time.sleep(0.1)
            loop_send += 1
        if r_id == 5:
            r_id = 1
        else:
            r_id += 1
    if enemy_color == 0:
        # 敌方为蓝方
        if flag:
            Port_operate.Map(b_id, np.float32(x), np.float(y), ser)
            time.sleep(0.1)
            loop_send += 1
        if b_id == 105:
            b_id = 101
        else:
            b_id += 1
    if nID == 4:
        if loop_send == 0:
            time.sleep(0.1)
        loop_send = 0
    nID = (nID + 1) % 5


def write(ser):
    while True:
        Map_Transmit(ser)
