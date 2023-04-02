import time
from collections import deque


class FpsCounter:
    """
    帧率计算器
    """
    def __init__(self, average_over=10):
        self._average_over = average_over
        self._time_queue = deque()
        self._time_queue.append(1)
        self._time_queue.append(2)
        self._fps = 0

    @property
    def fps(self) -> float:
        return self._fps

    def update(self):
        """
        更新帧率，每次调用都会当作是一帧并更新一次帧率
        会只计算最近 `average_over` 帧的平均帧率
        """
        if len(self._time_queue) > self._average_over:
            self._time_queue.popleft()
            self._fps = self._average_over / (self._time_queue[-1] - self._time_queue[0])
        else:
            self._fps = len(self._time_queue) / (self._time_queue[-1] - self._time_queue[0])
        self._time_queue.append(time.time())
