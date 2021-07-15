# Dataset utils and dataloaders

import glob
import math
import os
import random
import shutil
import time
from itertools import repeat
from multiprocessing.pool import ThreadPool
from pathlib import Path
from threading import Thread

import cv2
import numpy as np
import torch
from PIL import Image, ExifTags
from torch.utils.data import Dataset
from tqdm import tqdm


# Parameters
help_url = 'https://github.com/ultralytics/yolov5/wiki/Train-Custom-Data'
img_formats = ['bmp', 'jpg', 'jpeg', 'png', 'tif', 'tiff', 'dng']  # acceptable image suffixes
vid_formats = ['mov', 'avi', 'mp4', 'mpg', 'mpeg', 'm4v', 'wmv', 'mkv']  # acceptable video suffixes

def letterbox(img, height=608, width=1088,
              color=(127.5, 127.5, 127.5)):  # resize a rectangular image to a padded rectangular
    shape = img.shape[:2]  # shape = [height, width] = [1080, 1920]
    ratio = min(float(height) / shape[0], float(width) / shape[1]) # min(0.563, 0.567)
    new_shape = (round(shape[1] * ratio), round(shape[0] * ratio))  # new_shape = [width, height]，此时width/height=16/9
    dw = (width - new_shape[0]) / 2  # width padding
    dh = (height - new_shape[1]) / 2  # height padding
    top, bottom = round(dh - 0.1), round(dh + 0.1)
    left, right = round(dw - 0.1), round(dw + 0.1)
    img = cv2.resize(img, new_shape, interpolation=cv2.INTER_AREA)  # resized, no border

    # cv2.copyMakeBorder可以自动填充图像边界，cv2.BORDER_CONSTANT表示填充的像素值为常数，值为value
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)  # padded rectangular
    return img, ratio, dw, dh # (608, 1088, 3), 0.563, 3.5, 0.0

class LoadStreams:  # multiple IP or RTSP cameras
    def __init__(self, sources='streams.txt', img_size=(1088, 608)):

        if os.path.isfile(sources):
            with open(sources, 'r') as f:
                sources = [x.strip() for x in f.read().splitlines() if len(x.strip())]
        else:
            sources = [sources]

        n = len(sources)
        self.imgs = [None] * n
        self.sources = sources
        for i, s in enumerate(sources):
            # Start the thread to read frames from the video stream
            print('%g/%g: %s... ' % (i + 1, n, s), end='')

            self.cap = cv2.VideoCapture(eval(s) if s.isnumeric() else s)
            assert self.cap.isOpened(), 'Failed to open %s' % s


            self.vw = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.vh = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            self.frame_rate = int(round(self.cap.get(cv2.CAP_PROP_FPS)))

            if not s.isnumeric():
                self.vcnt = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))

            self.width = img_size[0]
            self.height = img_size[1]
            self.count = 0

            self.w, self.h = 1920, 1080

            _, self.imgs[i] = self.cap.read()  # guarantee first fram
            # self.imgs[i] = cv2.flip(self.imgs[i], 1)

            thread = Thread(target=self.update, args=([i, self.cap]), daemon=True)
            print(' success (%gx%g at %.2f FPS).' % (self.vw, self.vh, self.frame_rate))
            thread.start()

        # # check for common shapes
        # s = np.stack([letterbox(x, new_shape=self.img_size)[0].shape for x in self.imgs], 0)  # inference shapes
        # # print(len(self.imgs))
        # self.rect = np.unique(s, axis=0).shape[0] == 1  # rect inference if all shapes equal
        # if not self.rect:
        #     print('WARNING: Different stream shapes detected. For optimal performance supply similarly-shaped streams.')

    def update(self, index, cap):
        # Read next stream frame in a daemon thread
        n = 0
        while cap.isOpened():
            # n += 1
            # print(len(self.imgs))
            # _, self.imgs[index] = cap.read()
            # grab是指向下一个帧，retrieve是解码并返回一个帧
            cap.grab()
            # if n == 4:  # read every 4th frame
            _, self.imgs[index] = cap.retrieve()
            # self.imgs[index] = cv2.flip(self.imgs[index], 1)
            #n = 0
            time.sleep(1/self.frame_rate + 1)  # 等待时间（十分重要！）

    def __iter__(self):
        self.count = -1
        return self

    def __next__(self):
        self.count += 1

        img0 = self.imgs.copy() # (480, 640, 3)
        img = self.imgs.copy() # (480, 640, 3)

        for i, _ in enumerate(img0):
            img0[i] = cv2.resize(img0[i], (self.w, self.h))
            img[i], _, _, _ = letterbox(img0[i], height=self.height, width=self.width)
        
        img = np.stack(img, 0)
        img = img[:, :  , :, ::-1].transpose(0, 3, 1, 2)
        img = np.ascontiguousarray(img)
        img = img.astype(np.float32)
        img = np.divide(img, 255.0, out=img)
        # img /= 255.0

        # # Letterbox

        # img = [letterbox(x, new_shape=self.img_size, auto=self.rect)[0] for x in img0]
        # # Stack
        # img = np.stack(img, 0)

        # # Convert
        # img = img[:, :, :, ::-1].transpose(0, 3, 1, 2)  # BGR to RGB, to bsx3x416x416
        # img = np.ascontiguousarray(img)

        if cv2.waitKey(1) == ord('q'):  # q to quit
            cv2.destroyAllWindows()
            raise StopIteration

        return self.sources, img, img0 #, self.cap

    def __len__(self):
        return 0  # 1E12 frames = 32 streams at 30 FPS for 30 years
