from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *

class ImageViewer(QWidget): # 영상 frame viewer 클래스
    def __init__(self, parent=None):
        super(ImageViewer, self).__init__(parent)
        self.setFixedSize(640, 360) # 640*360은 원본 영상 1920*1080의 1/3 사이즈
        self.image = QImage()
        self.setAttribute(Qt.WA_OpaquePaintEvent)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.drawImage(0, 0, self.image)
        self.image = QImage()

    @pyqtSlot(QImage)
    def setImage(self, image):
        if image.isNull():
            print("Viewer Dropped frame!")

        self.image = image
        if image.size() != self.size():
            self.setFixedSize(image.size())
        self.update()