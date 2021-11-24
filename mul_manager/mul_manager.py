"""
多线程管理类
mul_manager.py
用于多线程间的信息发布与接收
"""

from queue import Queue
import threading


class MulManager(object):
    pub_list = []
    sub_list = []

    def create_pub(self, name):
        pub = MulPublisher()
        if pub.create(name, self.pub_list, self.sub_list):
            self.pub_list.append(pub)
            return pub
        else:
            return None

    def create_sub(self, name, num):
        sub = MulSubsriber()
        if sub.create(name, num, self.pub_list, self.sub_list):
            self.sub_list.append(sub)
            return sub
        else:
            return None


class MulPublisher(object):
    """
    publisher类
    """
    init_ok = False
    name = ""
    que = Queue(1)
    lock = threading.Lock()

    def __init__(self):
        pass

    def create(self, name, pub_list, sub_list):
        """
        name : string
        """
        if not self.check(name):
            print("名称应为string类型")
            return False

        if not self.link(name, pub_list, sub_list):
            print("已有相同名称的发布器")
            return False
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

    def link(self, name, pub_list, sub_list):
        for i in pub_list:
            if i.name == name:
                return False
        for i in sub_list:
            if i.name == name:
                print("已有接收器，正在匹配")
                self.lock = i.lock
                self.que = i.que
                return True
        print("没有找到接收器，自己生成锁")
        return True

    def pub(self, data):
        if not self.init_ok:
            print("未初始化成功,不能使用")
            return False
        self.lock.acquire()
        if self.que.full():
            self.que.get()
        self.que.put(data)
        self.lock.release()


class MulSubsriber(object):
    """
    subsriber类
    """
    init_ok = False
    name = ""
    que = Queue(1)
    lock = threading.Lock()

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

        if not self.link(name, pub_list, sub_list):
            print("已有相同名称的发布器")
            return False
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

    def link(self, name, pub_list, sub_list):
        for i in sub_list:
            if i.name == name:
                return False
        for i in pub_list:
            if i.name == name:
                print("已有发布器，正在匹配")
                self.lock = i.lock
                i.que = self.que
                return True
        print("没有找到发布器，使用自己生成的锁")
        return True

    def sub(self):
        if not self.init_ok:
            print("未初始化成功,不能使用")
            return None
        self.lock.acquire()
        if self.que.empty():
            re_data = None
        else:
            re_data = self.que.get()
        self.lock.release()
        return re_data
