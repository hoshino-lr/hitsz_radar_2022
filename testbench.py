from pyrdr.client import LiDARClient

ims = LiDARClient('tcp://127.0.0.1:8200')

while True:
    recv = ims.recv()
    print(recv)
