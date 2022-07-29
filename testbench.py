import os
import cv2 as cv

from pyrdr.server import ImageServer

# miku = cv.imread("/home/shiroki/Downloads/top_miku.png")
ims = ImageServer('tcp://*:5555')

input("Press Enter to continue...")

files = [f for f in os.listdir("/home/hoshino/下载/rmcvdata/roco_val/") if f.endswith(".jpg")]

for f in files:
    ims.send(cv.imread("/home/hoshino/下载/rmcvdata/roco_val/" + f))

