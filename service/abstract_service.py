"""
抽象线程，指定接口
created by 陈希峻 2023/3/20
"""

from abc import ABC, abstractmethod


class StartStoppableTrait(ABC):
    """
    所有线程的基类
    """
    @abstractmethod
    def start(self):
        return NotImplemented

    @abstractmethod
    def stop(self):
        return NotImplemented
