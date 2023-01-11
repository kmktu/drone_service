from PyQt5.QtWidgets import *
import sys
import init_ui as iu
from multiprocessing import freeze_support

class main_window(QMainWindow, QWidget):
    def __init__(self):
        super().__init__()
        self.init_ui()

    def init_ui(self):
        self.setWindowTitle("Drone Service")
        self.setMinimumSize(1920, 1080)

        wg = iu.init_layout()
        self.setCentralWidget(wg)
        self.show()

def main():
    app = QApplication(sys.argv)
    main_ = main_window()
    sys.exit(app.exec_())

if __name__ == '__main__':    
    # exe 파일로 만든 프로그램 실행시 Process 꺼짐을 방지하는 구문
    freeze_support()
    main()