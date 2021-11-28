"""
多线程管理类
mul_manager.py
用于多线程间的信息发布与接收
"""

from queue import Queue

pub_list = []
sub_list = []


class MulManager(object):
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
        if sub.create(name, num, pub_list,sub_list):
            sub_list.append(sub)
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

    def __init__(self):
        pass

    def create(self, name, pub_lists, sub_lists):
        """
        name : string
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

    # 已被抛弃的写法，queue自己有锁
    # def pub(self, data):
    #     if not self.init_ok:
    #         print("未初始化成功,不能使用")
    #         return False
    #     if self.lock.acquire(timeout=0.1):
    #         if self.que.full():
    #             # print("队列满")
    #             try:
    #                 self.que.get_nowait()
    #             except:
    #                 pass
    #         self.que.put(data)
    #         self.lock.release()
    #         return True
    #     else:
    #         print("获取锁失败")
    #         return False

    def pub(self, data):
        if not self.init_ok:
            print("未初始化成功,不能使用")
            return False
        if self.que.full():
            try:
                self.que.get_nowait()
            except:
                pass
        self.que.put(data)
        return True


class MulSubsriber(object):
    """
    subsriber类
    """
    init_ok = False
    name = ""
    que = Queue(1)

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

    # 已被抛弃的写法，queue自己有锁
    # def sub(self):
    #     if not self.init_ok:
    #         print("未初始化成功,不能使用")
    #         return None
    #     if self.lock.acquire(timeout=0.1):
    #         if self.que.empty():
    #             re_data = None
    #         else:
    #             re_data = self.que.get()
    #         self.lock.release()
    #     else:
    #         re_data = None
    #     return re_data

    def sub(self):
        if not self.init_ok:
            print("未初始化成功,不能使用")
            return None
        try:
            re_data = self.que.get_nowait()
        except:
            re_data = None
        return re_data
