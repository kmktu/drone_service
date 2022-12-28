from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import numpy as np
import cv2
import time

NAS_PATH = r"\\Di-nas-new\드론데이터\정제데이터"
VIDEO_PATH = r"D:\Download\Temp\2022-12-07_13-00_myongji_60_10_sunny_afternoon.mp4"

class VideoThread(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.running = True
        self.video_exit = True

    def run(self):
        video_path = "D:/drone/test2.mp4"

        self.cap = cv2.VideoCapture(VIDEO_PATH)
        print(self.running)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.signal.emit(frame)
                time.sleep(0.05)
            else:
                break

    def stop(self):
        self.cap.release()
        print("Video Release")

    def resume(self):
        self.running = True
        print("Video resume")

    def pause(self):
        self.running = False
        print("Video pause")


class VideoThread2(QThread):
    signal = pyqtSignal(np.ndarray)
    def __init__(self):
        super().__init__()
        self.running = True
        self.video_exit = True

    def run(self):
        video_path = "D:/drone/test2.mp4"
        self.cap = cv2.VideoCapture(VIDEO_PATH)
        while self.running:
            ret, frame = self.cap.read()
            if ret:
                self.signal.emit(frame)
                time.sleep(0.05)
            else:
                break

    def stop(self):
        self.cap.release()
        print("Video Release")

    def resume(self):
        self.running = True
        print("Video resume")

    def pause(self):
        self.running = False
        print("Video pause")

