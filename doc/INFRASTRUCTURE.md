# 雷达站 2022 架构

|        | 相机画面提供 | 车辆检测        | 点云      |
|--------|--------|-------------|---------|
| 比赛模式   | 相机驱动   | 运行          | 实时获取    |
| 测试神经网络 | 录像     | 运行          | 深度图？点云？ |
| 测试其他功能 | 录像     | 不运行，读取序列化数据 | 深度图？点云？ |

## 相机画面提供

### 相机驱动

需要主动取流

### 录像

整合读取序列化数据？（读序列化数据的时候显然也会用录像而不是实拍）

## 车辆检测

### 运行

没什么好说的

### 不运行，读取序列化数据

见录像

## 点云

### 实时获取


## 数据传递

从 Reactive 模型中抽出来最核心的元素：`on_next` 和 `on_completed`。为了贴合实际计算能力有限，可能发生过期数据堆积的场景，将被调用的 `on_next` 改为主动调用的 `latest`。相应的，`on_completed` 改为 `is_completed`。

```python
from typing import Callable


# ...

def spin(latest: Callable[[], str], is_completed: Callable[[], bool]):
    while not is_completed():
        data = latest()
        # do something with data, e.g.
        print(data)
```

- `latest` 会返回一个数据，如果没有数据可用则阻塞。相比使用管道，可以很方便地丢弃过期的数据，避免积压。
    - 存在潜在问题：假如 `is_completed` 设置的时机不对，那么 `latest` 会永久阻塞。用锁？用信号量？
- `is_completed` 用于判断是否已经没有数据可用了。

另外，也可以改变 `latest` 的返回值，使其返回一个 `Optional`，这样就不需要 `is_completed` 了。

```python
from typing import Callable, Optional


# ...

def spin(latest: Callable[[], Optional[str]]):
    while True:
        data = latest()
        if data is None:
            break
        # do something with data, e.g.
        print(data)
```

这种情况下，需要分析 `latest` 的实现（使用场景），确保它不会返回 `None`。

**重要：避免重复数据！**

引入标识符机制。`latest` 有一个额外的参数，由调用方提供。通过反转关系来避免包含对两边引用的对象分配出现。

### 实际范式

由线程包装对象持有该 Provider，并将它的 `latest` 暴露为一个“预先填入参数”的匿名函数（即柯里化）。

其中，这个“预先填入参数”的机制还可以用于自动生成自增标识符。考虑到“接线”全部在主函数中一次性完成，可以完美避免出现同一个接收器使用不同标识符的问题。