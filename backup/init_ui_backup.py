from PyQt5.QtWidgets import *
from PyQt5.QtGui import *
from PyQt5.QtCore import *
import cv2
import numpy as np
import backup.load_video_backup as lv
from yolov5 import yolo_detection
from concurrent.futures import ThreadPoolExecutor

from draw.draw_camera_groupbox import draw_camera_object_groupbox, draw_camera_action_groupbox
from draw.draw_month_barchart import draw_month_barchart
from draw.draw_file_list import draw_file_list

class init_layout(QWidget):
    def __init__(self):
        super().__init__()

        self.video_exist = False

        # load model
        self.inference_model_yolo = yolo_detection.ObjectDetection()

        # load video location widget
        self.location_1_label1 = QLabel(self)
        self.location_1_label1_start_btn = QPushButton("Start", self)
        self.location_1_label1_pause_btn = QPushButton("Pause", self)
        self.location_1_label1_stop_btn = QPushButton("Stop", self)
        self.location_1_label1.resize(1080, 960) # location to show video

        # test load video location widget
        self.location_1_label2 = QLabel(self)
        self.location_1_label2_btn_1 = QPushButton("start")
        self.location_1_label2_btn_2 = QPushButton("Stop")
        self.location_1_label2_btn_3 = QPushButton("Disconnect")
        self.location_1_label2.resize(1080, 960)

        main_layout = QVBoxLayout()

        load_video_layout = QVBoxLayout()
        play_video_layout = QHBoxLayout()
        play_video_btn_layout = QHBoxLayout()

        information_widget = QWidget()
        information_widget.setFixedHeight(300)
        information_layout = QHBoxLayout()
        information_each_camera_detection_layout = QVBoxLayout()
        information_each_camera_action_layout = QVBoxLayout()
        information_monthly_detecting_layout = QVBoxLayout()
        information_area_detecting_layout = QVBoxLayout()

        # Load Video first layout
        play_video_layout.addWidget(self.location_1_label1)
        play_video_layout.addWidget(self.location_1_label2)
        play_video_btn_layout.addWidget(self.location_1_label1_start_btn)
        play_video_btn_layout.addWidget(self.location_1_label1_pause_btn)
        play_video_btn_layout.addWidget(self.location_1_label1_stop_btn)

        # Load Video list
        self.file_list_widget = draw_file_list()
        self.file_list_widget.currentItemChanged.connect(self.chk_current_item_changed)
        play_video_layout.addWidget(self.file_list_widget)

        # Load Video second layout

        # each camera object detecting information
        self.camera_object_groupbox_widget = draw_camera_object_groupbox()
        information_each_camera_detection_layout.addWidget(self.camera_object_groupbox_widget)

        # each camera anomaly action information
        self.camera_action_groupbox_widget = draw_camera_action_groupbox()
        information_each_camera_action_layout.addWidget(self.camera_action_groupbox_widget)

        # Monthly detecting information
        self.month_barchart_widget = draw_month_barchart()
        information_monthly_detecting_layout.addWidget(self.month_barchart_widget)

        # Click button event
        self.location_1_label1_start_btn.clicked.connect(self.location_1_label1_start_btn_clicked)
        self.location_1_label1_pause_btn.clicked.connect(self.location_1_label1_pause_btn_clicked)
        self.location_1_label1_stop_btn.clicked.connect(self.location_1_label1_stop_btn_clicked)

        # self.location_1_label2_btn_1.clicked.connect(self.location_1_label2_btn_1_clicked_event)
        # self.location_1_label2_btn_2.clicked.connect(self.location_1_label2_btn_2_clicked_event)

        # set load video layout
        load_video_layout.addLayout(play_video_layout)
        load_video_layout.addLayout(play_video_btn_layout)

        # set information layout
        information_layout.addLayout(information_each_camera_detection_layout)
        information_layout.addLayout(information_each_camera_action_layout)
        information_layout.addLayout(information_monthly_detecting_layout)
        information_layout.addLayout(information_area_detecting_layout)

        main_layout.addLayout(load_video_layout)
        information_widget.setLayout(information_layout)
        main_layout.addWidget(information_widget)

        self.setLayout(main_layout)

        self.video_thread1 = lv.VideoThread()
        self.video_thread2 = lv.VideoThread2()

        # thread connect to Qlabel
        self.video_thread1.signal.connect(self.update_image)
        self.video_thread2.signal.connect(self.update_image2)

    @pyqtSlot(np.ndarray)
    def update_image(self, frame):
        qt_img = self.convert_cv_qt(frame)
        self.location_1_label1.setPixmap(qt_img)

    @pyqtSlot(np.ndarray)
    def update_image2(self, frame):
        print("")
        executor = ThreadPoolExecutor(1)
        executor.submit(self.inference_model_yolo.inference_img, frame)
        inference_img = self.inference_model_yolo.get_data()
        inference_qt_imt = self.convert_cv_qt(inference_img)
        self.location_1_label2.setPixmap(inference_qt_imt)
        # with ThreadPoolExecutor(1) as executor:
        #     inference_img = executor.map(self.inference_model_yolo.inference_img, frame)
    # t = threading.Thread(target=self.inference_model_yolo.inference_img, args=(frame))
    # t.start()

    def convert_cv_qt(self, frame):
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        p = convert_to_Qt_format.scaled(640, 480, Qt.IgnoreAspectRatio)
        return QPixmap.fromImage(p)

    def location_1_label1_start_btn_clicked(self):
        if not self.video_exist:
            self.video_thread1.start()
            self.video_thread2.start()
            self.video_exist = True
        else:
            self.video_thread1.resume()
            self.video_thread2.resume()

    def location_1_label1_pause_btn_clicked(self):
        self.video_thread1.pause()
        self.video_thread2.pause()

    def location_1_label1_stop_btn_clicked(self):
        self.video_thread1.stop()
        self.video_thread2.stop()

    def chk_current_item_changed(self):
        print("Clicked Video : " + self.file_list_widget.currentItem().text())