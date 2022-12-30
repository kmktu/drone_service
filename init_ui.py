import cv2
import load_video.load_video as lv
from concurrent.futures import ThreadPoolExecutor

from multiprocessing import Process, Queue

from draw.draw_camera_groupbox import draw_camera_object_groupbox, draw_camera_action_groupbox
from draw.draw_month_barchart import draw_month_barchart
from draw.draw_file_list import draw_file_list
from load_video.ImageViewer import *
import time


"""Dev Options"""
video_sync = False  # Sync 기능 On/Off


class init_layout(QWidget):
    def __init__(self): # UI 초기화
        super().__init__()
        self.video_load = False # video 로드 여부
        self.video_play = False # video 재생 여부
        self.last_video = "" # 마지막 재생 영상 기록

        self.original_video = ImageViewer() # 원본 영상 Viewer
        self.detected_video = ImageViewer() # 추론 영상 Viewer

        # ADD
        self.recognize_video = ImageViewer() # action Viewer

        self.video_start_btn = QPushButton("Start", self) # 영상 재생 버튼
        self.video_start_btn.clicked.connect(self.video_start)
        self.video_pause_btn = QPushButton("Pause", self) # 영상 일시정지 버튼
        self.video_pause_btn.clicked.connect(self.video_pause)
        self.video_stop_btn = QPushButton("Stop", self) # 영상 초기화 버튼
        self.video_stop_btn.clicked.connect(self.video_stop)

        main_layout = QVBoxLayout() # 메인 레이아웃(비디오영역, 비디오 컨트롤러 영역, 통계 영역)
        play_video_layout = QHBoxLayout() # 비디오 레이아웃(원본 영상, 추론 영상, 영상 리스트)
        play_video_btn_layout = QHBoxLayout() # 비디오 플레이어 레이아웃(재생버튼, 일시정지 버튼, 초기화 버튼)

        information_widget = QWidget()
        information_widget.setFixedHeight(300) # 높이 300 고정
        information_layout = QHBoxLayout() # 통계 레이아웃(객체 통계, 액션 통계, 월별 통계)

        play_video_layout.addWidget(self.original_video) # 원본 영상
        play_video_layout.addWidget(self.detected_video) # 추론 영상

        # ADD
        play_video_layout.addWidget(self.recognize_video) # action Video

        play_video_btn_layout.addWidget(self.video_start_btn) # 영상 재생 버튼
        play_video_btn_layout.addWidget(self.video_pause_btn) # 영상 일시정지 버튼
        play_video_btn_layout.addWidget(self.video_stop_btn) # 영상 초기화 버튼

        self.file_list_widget = draw_file_list() # 파일 리스트 불러오기(listwidget 리턴)
        self.file_list_widget.currentItemChanged.connect(self.chk_current_item_changed) # listwidget 아이템 선택시 이벤트
        play_video_layout.addWidget(self.file_list_widget)

        self.camera_object_groupbox_widget = draw_camera_object_groupbox() # 객체 통계 그룹박스 불러오기(QGroupBox 리턴)
        information_layout.addWidget(self.camera_object_groupbox_widget)

        self.camera_action_groupbox_widget = draw_camera_action_groupbox() # 액션 통계 그룹박스 불러오기(QGroupBox 리턴)
        information_layout.addWidget(self.camera_action_groupbox_widget)

        self.month_barchart_widget = draw_month_barchart() # 월별 통계 barchart 불러오기(barchar 리턴)
        information_layout.addWidget(self.month_barchart_widget)

        main_layout.addLayout(play_video_layout)
        main_layout.addLayout(play_video_btn_layout)
        information_widget.setLayout(information_layout)
        main_layout.addWidget(information_widget)

        self.setLayout(main_layout)

        self.vis1_ready = False
        self.vis2_ready = False
        self.vis_terminate = False

        # ADD model init
        # self.model_init = lv.LoadVideo_model()
        # print("model init !!!")
        # self.inference_model_yolo, self.inference_model_slowfast = lv.model_init()

    def video_start(self): # 영상 재생 함수
        if self.file_list_widget.currentItem() == None: # listwidget 아이템 미선택시 바로 리턴
            return
        else:
            video_path = self.file_list_widget.currentItem().text()  # listwidget으로부터 선택된 영상 경로 불러오기
            if self.last_video != video_path: # 마지막 재생 영상과 현재 선택된 영상 미일치시 비디오 처음부터 다시 재생
                self.last_video = video_path
                if self.video_load == True:
                    self.video_stop()

            if self.video_load: # 영상 일시정지 상태에서 다시 재생시
                self.video_play = True
            else:               # 초기 영상 Loading
                self.video_load = True
                self.video_play = True
                self.vis_terminate = False

                self.frame_q = Queue()  # 선택된 영상의 frame이 담길 Queue
                self.detect_q = Queue() # 추론된 영상의 frame이 담길 Queue

                # ADD 행동 큐
                self.action_detect_q = Queue() # 행동 추론 영상의 frame이 담길 Queue

                self.frame_reader_p = Process(target=lv.read_frames, args=(self.frame_q, self.detect_q, video_path),
                                              name="READ_FRAME_P") # 쓰레드를 통한 영상 프레임 읽기(원본 영상 프레임은 frame_q에, 추론 영상 프레임은 detect_q에 쌓임)

                # self.frame_reader_p = Process(target=self.model_init.read_frames, args=(self.frame_q, self.detect_q,
                #                                                                           video_path),
                #                               name="READ_FRAME_P")

                executor = ThreadPoolExecutor(1) # 쓰레드를 통한 원본 영상 및 추론 영상 프레임 가시화
                executor.submit(self.visual_process)

                # ADD 행동 프로세스
                self.frame_reader_p2 = Process(target=lv.slowfast_read_frames, args=(self.frame_q, self.action_detect_q,
                                                                                     video_path),
                                               name="SLOWFAST_FRAME_P")
                # self.frame_reader_p2 = Process(target=self.load_model.slowfast_read_frames, args=(self.action_detect_q,
                #                                                                                   video_path),
                #                                name="SLOWFAST_FRAME_P")
                #
                executor2 = ThreadPoolExecutor(1)  # 쓰레드를 통한 원본 영상 및 추론 영상 프레임 가시화
                executor2.submit(self.visual_process2)

                self.frame_reader_p.daemon = True
                self.frame_reader_p.start()

                self.frame_reader_p2.daemon = False
                self.frame_reader_p2.start()

                print("start")

    def video_pause(self): # 영상 일시정지 함수
        self.video_play = False
        print("pause")

    def video_stop(self): # 영상 초기화 함수
        self.frame_q = None
        self.detect_q = None
        self.action_detect_q = None
        self.video_load = False
        self.video_play = False
        if self.frame_reader_p.is_alive():
            self.frame_reader_p.terminate()
        if self.frame_reader_p2.is_alive():
            self.frame_reader_p2.terminate()
        self.vis_terminate = True
        self.vis1_ready = False
        self.vis2_ready = False
        print("stop")

    def cls_count(self, detect_count_dict: dict):  # Object Count 갱신 함수
        # 클래스 정의 (리소스 낭비로 인해 이 위치 외에 옮길 좋은 위치 필요)
        actions = ['sos', 'fall_down']
        objects = ['car', 'person', 'boat']

        # 최신 개수 받아오기
        total_object_count = self.camera_object_groupbox_widget.findChild(QLabel, 'total_object_count')
        person_count = self.camera_object_groupbox_widget.findChild(QLabel, 'person_count')
        car_count = self.camera_object_groupbox_widget.findChild(QLabel, 'car_count')
        boat_count = self.camera_object_groupbox_widget.findChild(QLabel, 'boat_count')
        total_action_count = self.camera_action_groupbox_widget.findChild(QLabel, 'total_action_count')
        sos_count = self.camera_action_groupbox_widget.findChild(QLabel, 'sos_count')
        fall_down_count = self.camera_action_groupbox_widget.findChild(QLabel, 'fall_down_count')

        # detect_count_dict에 포함된 Object 개수 더하기
        total_obj_cnt = int(total_object_count.text())
        total_act_cnt = int(total_action_count.text())
        for cls, value in detect_count_dict.items():
            if cls in actions or cls in objects:
                cnt = int(eval(f'{cls}_count.text()'))
                if cls in objects:
                    cnt += value
                    total_obj_cnt += value
                else:
                    cnt += 1
                    total_act_cnt += 1
                eval(f"{cls}_count.setText(f'{{cnt}}')")
        total_object_count.setText(f'{total_obj_cnt}')
        total_action_count.setText(f'{total_act_cnt}')

    def visual_process(self): # 영상 가시화 함수
        while True:
            self.vis1_ready = False
            if self.frame_q.qsize() > 0 and self.video_play: # 영상이 재생 중이며 frame_q에 frame이 하나 이상 존재할 때 가시화
                frame = self.frame_q.get()
                frame = self.convert_cv_qt(frame)
                detect_result = self.detect_q.get()
                detect_frame = self.convert_cv_qt(detect_result[0])  # detect_q 중 frame
                self.vis1_ready = True
                while not self.vis2_ready and video_sync:
                    if self.vis1_ready and self.vis2_ready or self.vis_terminate:
                        break
                self.original_video.setImage(frame)
                self.detected_video.setImage(detect_frame)
                detect_count = detect_result[1]  # detect_q 중 라벨 수 dict
                self.cls_count(detect_count)
                time.sleep(0.04)
            else:
                continue

    # ADD action recognize visualization
    def visual_process2(self): # 영상 가시화 함수
        action_detect_q_flag = False
        while True:
            self.vis2_ready = False
            if self.action_detect_q.qsize() > 0:
                action_detect_q_flag = True
                self.recognize_frame = self.action_detect_q.get()
                if type(self.recognize_frame) == dict:
                    ## 행동 confidence 값
                    # 이거 이용해서 count된 값 보여주면 될듯
                    action_count = self.recognize_frame
                    self.cls_count(action_count)
                else:
                    self.recognize_frame = self.convert_cv_qt(self.recognize_frame)

            elif self.action_detect_q.qsize() == 0:
                action_detect_q_flag = False

            if action_detect_q_flag:
                if type(self.recognize_frame) != dict:
                    self.vis2_ready = True
                    while not self.vis1_ready and video_sync:
                        if self.vis1_ready and self.vis2_ready or self.vis_terminate:
                            break
                    self.recognize_video.setImage(self.recognize_frame)
                    time.sleep(0.04)
            else:
                continue

    def convert_cv_qt(self, frame): # cv2 이미지를 QImage 형태로 변환하는 함수(640*360 사이즈로 자동 변환)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(640, 360, Qt.IgnoreAspectRatio)
        return scaled_img

    def chk_current_item_changed(self): # listwidget 아이템 선택시 발생하는 이벤트
        print("Clicked Video : " + self.file_list_widget.currentItem().text())