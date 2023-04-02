from loguru import logger

import zmq
import cv2 as cv
import numpy as np

from proto.rdr.detected_armor_pb2 import ImageAndArmor, CarInfo
from proto.rdr.lidar_pb2 import LiDARDepthPixels
from proto.rdr.encoded_img_pb2 import EncodedImg


class ImageClient:
    def __init__(self, endpoint: str, context=None):
        self.context = context or zmq.Context.instance()
        self.socket: zmq.Socket = self.context.socket(zmq.constants.SocketType.SUB)
        self.socket.connect(endpoint)
        self.socket.subscribe(b'')
        # self.socket.RCVTIMEO = 1000

        self.poller = zmq.Poller()
        self.poller.register(self.socket, zmq.POLLIN)
        # self.socket.close()

    def recv(self, timeout=1000) -> np.ndarray:
        # data = self.socket.recv()
        ok = self.poller.poll(timeout)
        if not ok:
            return None
        data = self.socket.recv()
        # print(data)
        # data = data[0]
        result: EncodedImg = EncodedImg.FromString(data)

        logger.debug(f'Received armor @{result.timestamp}')
        img = cv.imdecode(np.frombuffer(result.data, np.uint8), cv.IMREAD_UNCHANGED)
        return img


class ImageAndArmorClient:
    def __init__(self, endpoint: str, context=None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.constants.SocketType.SUB)
        self.socket.connect(endpoint)
        self.socket.subscribe(b'')

    def recv(self) -> (np.ndarray, list[CarInfo]):
        data = self.socket.recv()
        # print(len(data))
        result: ImageAndArmor = ImageAndArmor.FromString(data)
        # print(str(result))
        logger.debug(f'Received armor @{result.timestamp}')
        # print(result.armors)
        # print(result.timestamp)
        # np.ndarray
        img = cv.imdecode(np.frombuffer(result.data, np.uint8), cv.IMREAD_UNCHANGED)
        return img, list(result.armors)


class LiDARClient:
    def __init__(self, endpoint: str, context=None):
        self.context = context or zmq.Context.instance()
        self.socket = self.context.socket(zmq.constants.SUB)
        self.socket.connect(endpoint)
        self.socket.subscribe(b'')

    def recv(self) -> list[tuple[int, int, float]]:
        data = self.socket.recv()
        result: LiDARDepthPixels = LiDARDepthPixels.FromString(data)
        logger.debug(f'Received lidar @{result.timestamp}')
        return [(p.x, p.y, p.z) for p in result.pixels]
