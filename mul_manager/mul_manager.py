"""
多线程管理类
mul_manager.py
用于多线程间的信息发布与接收
created by 李龙 
"""
import time
from queue import Queue
from threading import Event

pub_list = []
sub_list = []


class MulManager(object):
    """
    多线程管理类
    负责多线程之间的通讯和链接
    """
    global pub_list
    global sub_list

    def create_pub(self, name):
        pub = MulPublisher()
        if pub.create(name, pub_list, sub_list):
            pub_list.append(pub)
            return pub
        else:
            return None

    def create_sub(self, name, num):
        sub = MulSubsriber()
        if sub.create(name, num, pub_list, sub_list):
            sub_list.append(sub)
            return sub
        else:
            return None


class MulPublisher(object):
    """
    publisher类
    :param init_ok
    :param
    """
    init_ok = False
    name = ""

    def __init__(self):
        """
        :param none
        """
        self.que = Queue()
        self.event = Event()
        pass

    def create(self, name, pub_lists, sub_lists):
        """
        :param name : string
        :param publists
        :param sublists
        """
        if not self.check(name):
            print("名称应为string类型")
            return False
        for i in pub_lists:
            if i.name == name:
                print("已有相同名称的发布器")
                return False
        for i in sub_lists:
            if i.name == name:
                print("已有队列，正在匹配")
                self.que = i.que
                self.event = i.event
        print("初始化成功")
        self.name = name
        self.init_ok = True
        return True

    def release(self):
        self.init_ok = False
        self.que = Queue(1)
        pass

    def check(self, name):
        if not isinstance(name, str):
            return False
        else:
            return True

    def pub(self, data):
        if not self.init_ok:
            print("未初始化成功,不能使用")
            return
        if self.event.is_set():
            return
        if self.que.full():
            try:
                self.que.get_nowait()
            except:
                pass
        self.que.put_nowait(data)
        self.event.set()


class MulSubsriber(object):
    """
    subsriber类
    """
    init_ok = False
    name = ""
    que = Queue(1)
    event = Event()

    def __init__(self):
        pass

    def create(self, name, queue_num, pub_list, sub_list):
        """
        name : string
        queue_num : 队列大小
        """
        self.que = Queue(queue_num)
        if not self.check(name):
            print("名称应为string类型")
            return False
        for i in sub_list:
            if i.name == name:
                print("已有相同名称的发布器")
                return False
        for i in pub_list:
            if i.name == name:
                print("已有发布器，正在匹配")
                i.que = self.que
                self.event = i.event
        print("初始化成功")
        self.name = name
        self.init_ok = True
        return True

    def release(self):
        self.init_ok = False
        self.que = Queue(1)
        pass

    def check(self, name):
        if not isinstance(name, str):
            return False
        else:
            return True

    def sub(self):
        if not self.init_ok:
            print("未初始化成功,不能使用")
        self.event.wait()
        try:
            re_data = self.que.get_nowait()
        except:
            re_data = None
        self.event.clear()
        return re_data
