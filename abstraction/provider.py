"""
关于 `latest` 的约定，请参考 `doc/INFRASTRUCTURE.md`
"""
from abc import ABCMeta
from threading import Event
from typing import TypeVar, Generic, Optional
import unittest
from loguru import logger

T = TypeVar('T')


class GenericProvider(Generic[T], metaclass=ABCMeta):
    def __init__(self):
        self._data = None
        self._is_latest_set = set()
        self._is_end = False
        self._push_or_end_event = Event()
        # self._last_push_time = 0   # 上一次推送数据的时间（不用的原因是需要在 `push` 里面加入额外设置代码
        # self._probably_dead = False  # 是否可能死掉了，由是否超时判断（不用的原因是似乎让后面的 spin 变快毫无意义

    def push(self, new_data: T):
        """
        推送数据
        :param new_data: 要推送的数据
        :return: None
        """
        self._data = new_data
        self._is_latest_set.clear()
        self._push_or_end_event.set()

    def end(self):
        """
        结束数据流
        :return: None
        """
        self._is_end = True
        self._is_latest_set.clear()
        self._push_or_end_event.set()

    def latest(self,
               timeout: float | None = None,
               identifier=None
               ) -> Optional[T]:
        """
        获取最新的数据。如果数据已经被获取过，那么阻塞；如果发生错误（当前可能的错误有：超时），返回 None
        :param timeout: None 表示无限等待，0 表示不等待，其他表示等待的时间
        :param identifier: 用于标识的标识符，如果不传入，则不会阻塞
        :return: 数据或者 None
        """
        if self._is_end:
            return None

        # 如果标识符显示已经获取过了，那么阻塞等待更新的数据，注意这个函数阻塞的是接收方线程
        if timeout is not None and identifier in self._is_latest_set:
            # logger.info(f"latest: {identifier} is waiting for data update, timeout: {timeout}")
            result = self._push_or_end_event.wait(timeout / 1000)  # 阻塞等待数据更新，添加 timeout 避免前面线程挂掉的时候无限阻塞
            # print(f"latest: {result}")
            if self._is_end:
                return None

        # 手动 clear 确保有阻塞下次调用的能力，是否说明应该直接用 Condition？
        self._push_or_end_event.clear()

        if identifier is not None:
            self._is_latest_set.add(identifier)
        return self._data


class NullProvider(GenericProvider[T]):
    def __init__(self):
        super().__init__()
        self._data = None

    def push(self, new_data: T):
        pass

    def end(self):
        pass

    def latest(self, timeout: int | None = None, identifier=None) -> Optional[T]:
        return None


class ProviderTest(unittest.TestCase):
    def test_provider(self):
        import threading

        provider = GenericProvider[int]()
        provider.push(1)
        provider.push(2)
        self.assertEqual(provider.latest(), 2)
        self.assertEqual(provider.latest(), 2)

        ident = "random identifier"
        self.assertEqual(provider.latest(identifier=ident), 2)

        thread = threading.Thread(
            target=lambda: self.assertEqual(
                provider.latest(identifier=ident), 3))
        thread.start()
        provider.push(3)
        thread.join()

        provider.end()
        self.assertIsNone(provider.latest())


if __name__ == '__main__':
    unittest.main()
