"""
串口通信
"""
from .official import official_Judge_Handler, Game_data_define
from .port_operation import Port_operate

buffer = [0] * 50
bufferbyte = 0
cmd_id = 0

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
    cmd_use = [0x0001, 0x0002, 0x0003, 0x0101, 0x0105, 0x0301, 0x0303]
    bufferbyte = 0
    while True:
        s = int().from_bytes(ser.read(1), 'big')

        if bufferbyte >= 50:
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
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 41):
                # 统计血量
                Port_operate.Robot_HP(buffer)
                read_init(buffer)
                continue

        # 场地事件数据
        if bufferbyte == 12 and cmd_id == 0x0101:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 13):
                Port_operate.Receive_State_Data(buffer)
                read_init(buffer)
                continue

        # 飞镖发射倒计时
        if bufferbyte == 9 and cmd_id == 0x0105:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 10):
                Game_dart_remaining_time.time = buffer[7]
                read_init(buffer)
                continue

        # 小地图通信
        if bufferbyte == 24 and cmd_id == 0x0303:
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 24):
                # 此处加入处理
                read_init(buffer)
                continue

        # 云台手通信
        if bufferbyte == 19 and cmd_id == 0x0301:  # 2bite数据
            if official_Judge_Handler.myVerify_CRC16_Check_Sum(id(buffer), 19):
                # 比赛阶段信息
                Port_operate.Receive_Robot_Data(buffer)
                read_init(buffer)
                continue
        bufferbyte += 1


def write(ser):
    Port_operate.port_send_init()
    while True:
        Port_operate.port_send(ser)
