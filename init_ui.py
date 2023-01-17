import cv2
import load_video.load_video as lv
from concurrent.futures import ThreadPoolExecutor
from multiprocessing import Process, Queue, Pipe
from draw.draw_camera_groupbox import draw_camera_object_groupbox, draw_camera_action_groupbox
from load_video.ImageViewer import *
import time
import os
from PyQt5.QtCore import Qt
import pyqtgraph as pg
import json
from collections import defaultdict, OrderedDict

class init_layout(QWidget):
    def __init__(self): # UI 초기화
        super().__init__()
        self.video_load = False # video 로드 여부
        self.video_play = False # video 재생 여부
        self.last_video = "" # 마지막 재생 영상 기록

        self.video_sync = True

        self.original_video = ImageViewer() # 원본 영상 Viewer
        self.detected_video = ImageViewer() # 추론 영상 Viewer
        self.recognize_video = ImageViewer() # action Viewer

        self.video_start_btn = QPushButton("Start", self) # 영상 재생 버튼
        self.video_start_btn.clicked.connect(self.video_start)
        self.video_pause_btn = QPushButton("Pause", self) # 영상 일시정지 버튼
        self.video_pause_btn.clicked.connect(self.video_pause)
        self.video_stop_btn = QPushButton("Stop", self) # 영상 초기화 버튼
        self.video_stop_btn.clicked.connect(self.video_stop)
        self.reload_plot_data_btn = QPushButton("Refresh", self)  # 통계 불러오기 버튼
        self.reload_plot_data_btn.clicked.connect(self.reload_plot_data)

        # 파일리스트 경로 지정 버튼
        self.video_file_list_btn = QPushButton("ADD File List Path", self)
        self.video_file_list_btn.clicked.connect(self.video_file_list_path)

        # 버튼 폰트 사이즈 크기 설정
        font_size = 12
        start_btn_font = self.video_start_btn.font()
        start_btn_font.setPointSize(font_size)
        self.video_start_btn.setFont(start_btn_font)

        stop_btn_font = self.video_stop_btn.font()
        stop_btn_font.setPointSize(font_size)
        self.video_stop_btn.setFont(stop_btn_font)

        pause_btn_font = self.video_pause_btn.font()
        pause_btn_font.setPointSize(font_size)
        self.video_pause_btn.setFont(pause_btn_font)

        file_list_btn_font = self.video_file_list_btn.font()
        file_list_btn_font.setPointSize(font_size)
        self.video_file_list_btn.setFont(file_list_btn_font)

        reload_plot_data_btn_font = self.reload_plot_data_btn.font()
        reload_plot_data_btn_font.setPointSize(font_size)
        self.reload_plot_data_btn.setFont(reload_plot_data_btn_font)

        main_layout = QVBoxLayout() # 메인 레이아웃(비디오영역, 비디오 컨트롤러 영역, 통계 영역)

        all_play_video_layout = QVBoxLayout() # 전체 비디오 레이아웃(영상, 라벨)
        video_qlabel_layout = QHBoxLayout()
        video_sync_layout = QHBoxLayout()   # 비디오 싱크 레이아웃
        video_top_line_layout = QVBoxLayout()
        play_video_label_layout = QHBoxLayout() # 비디오 라벨 레이아웃(라벨)
        play_video_layout = QHBoxLayout() # 비디오 레이아웃(원본 영상, 추론 영상)

        all_file_list_layout = QVBoxLayout() # 전체 파일 리스트 레이아웃(파일리스트, 버튼)
        file_list_btn_layout = QHBoxLayout() # 파일리스트 버튼 레이아웃(버튼)
        file_list_layout = QHBoxLayout() # 파일 리스트 레이아웃

        empty_widget = QWidget()        # 영상 부분과 파일리스트 사이를 띄우기 위함 공백임
        empty_widget.setFixedHeight(15)
        empty_box_layout = QHBoxLayout()
        empty_box_layout.addWidget(empty_widget)

        empty_widget2 = QWidget()  # 파일리스트와 통계부분 띄우기 위함
        empty_widget2.setFixedHeight(15)
        empty_box_layout2 = QHBoxLayout()
        empty_box_layout2.addWidget(empty_widget)

        play_video_btn_layout = QHBoxLayout() # 비디오 플레이어 레이아웃(재생버튼, 일시정지 버튼, 초기화 버튼)

        all_information_layout = QVBoxLayout()

        information_widget = QWidget()
        information_widget.setFixedHeight(300) # 높이 300 고정
        information_layout = QHBoxLayout() # 통계 레이아웃(객체 통계, 액션 통계, 월별 통계)
        information_label_layout = QHBoxLayout()  # 통계 제목 레이아웃(통계 제목, 정보 새로 고침 버튼)

        # 영상, 영상 라벨 전체 부분
        self.play_video_qlabel = QLabel(self)
        self.play_video_qlabel.setFont(QFont('Arial', 15))
        self.play_video_qlabel.setText("Video")
        self.play_video_qlabel.setFixedSize(68, 20)

        # UI 상태 표시 부분
        self.model_init_log = QLabel(self)
        self.model_init_log.setFont(QFont('Arial', 10))
        self.model_init_log.setText("State : Waiting...")
        # self.model_init_log.setFixedSize(68, 20)

        # FPS 상태 표시 부분
        video_fps_log_box = QGroupBox('FPS')
        video_fps_log_box_layout = QHBoxLayout()
        self.fps_widget_label = QLabel()
        self.obj_fps_widget_label = QLabel()
        self.act_fps_widget_label = QLabel()
        self.fps_widget_label.setFont(QFont('Arial', 10))
        self.obj_fps_widget_label.setFont(QFont('Arial', 10))
        self.act_fps_widget_label.setFont(QFont('Arial', 10))

        video_fps_log_box_layout.addWidget(self.fps_widget_label)
        video_fps_log_box_layout.addWidget(self.obj_fps_widget_label)
        video_fps_log_box_layout.addWidget(self.act_fps_widget_label)
        video_fps_log_box.setLayout(video_fps_log_box_layout)

        self.video_sync_qlabel = QLabel(self)
        self.video_sync_qlabel.setFont(QFont('Arial', 13))
        self.video_sync_qlabel.setText("Video Sync")
        self.video_sync_qlabel.setFixedSize(100, 20)

        all_play_video_line1 = QFrame()
        all_play_video_line1.setFrameShape(QFrame.HLine)
        all_play_video_line1.setFrameShadow(QFrame.Sunken)

        # Qlabel 레이아웃 위젯
        video_qlabel_layout.addWidget(self.play_video_qlabel, alignment=Qt.AlignLeft)
        video_qlabel_layout.addWidget(self.model_init_log, alignment=Qt.AlignCenter)
        video_qlabel_layout.addWidget(video_fps_log_box, alignment=Qt.AlignRight)

        # 영상 싱크 맞추기 레이아웃 위젯
        self.radio_box_sync = QGroupBox()
        self.radio_box_sync_layout = QHBoxLayout()
        self.radio_box_sync.setLayout(self.radio_box_sync_layout)
        self.radio_box_sync_layout.addWidget(self.video_sync_qlabel)
        self.btn_video_sync_true = QRadioButton('Yes')
        self.btn_video_sync_true.setChecked(True)
        self.btn_video_sync_true.clicked.connect(self.video_sync_clicked)
        self.radio_box_sync_layout.addWidget(self.btn_video_sync_true)

        self.btn_video_sync_false = QRadioButton('No')
        self.btn_video_sync_false.clicked.connect(self.video_sync_clicked)
        self.radio_box_sync_layout.addWidget(self.btn_video_sync_false)
        video_qlabel_layout.addWidget(self.radio_box_sync, alignment=Qt.AlignRight)

        # 영상 부분 라인 그리기
        video_top_line_layout.addWidget(all_play_video_line1)

        # 영상 부분 라인 그리기
        play_video_line2 = QFrame()
        play_video_line3 = QFrame()
        play_video_line2.setFrameShape(QFrame.VLine)
        play_video_line2.setFrameShadow(QFrame.Sunken)
        play_video_line3.setFrameShape(QFrame.VLine)
        play_video_line3.setFrameShadow(QFrame.Sunken)

        # 영상 부분 위젯
        play_video_layout.addWidget(self.original_video) # 원본 영상
        play_video_layout.addWidget(play_video_line2)
        play_video_layout.addWidget(self.detected_video) # 추론 영상
        play_video_layout.addWidget(play_video_line3)
        play_video_layout.addWidget(self.recognize_video) # action Video

        # 영상 라벨 위젯
        self.play_video_label1 = QLabel(self)
        self.play_video_label1.setFont(QFont('Arial', 12))
        self.play_video_label1.setText("Original Video")
        self.play_video_label1.setAlignment(Qt.AlignCenter)

        self.play_video_label2 = QLabel(self)
        self.play_video_label2.setFont(QFont('Arial', 12))
        self.play_video_label2.setText("Object Detection")
        self.play_video_label2.setAlignment(Qt.AlignCenter)

        self.play_video_label3 = QLabel(self)
        self.play_video_label3.setFont(QFont('Arial', 12))
        self.play_video_label3.setText("Action Detection")
        self.play_video_label3.setAlignment(Qt.AlignCenter)

        play_video_label_layout.addWidget(self.play_video_label1)
        play_video_label_layout.addWidget(self.play_video_label2)
        play_video_label_layout.addWidget(self.play_video_label3)

        # 전체 파일리스트 위젯
        self.file_list_qlabel = QLabel(self)
        self.file_list_qlabel.setFont(QFont('Arial', 15))
        self.file_list_qlabel.setText("File List")
        self.file_list_qlabel.setFixedSize(68, 20)

        all_file_list_line1 = QFrame()
        all_file_list_line1.setFrameShape(QFrame.HLine)
        all_file_list_line1.setFrameShadow(QFrame.Sunken)

        all_file_list_layout.addWidget(self.file_list_qlabel)
        all_file_list_layout.addWidget(all_file_list_line1)

        # 파일리스트 위젯
        self.file_list_widget = QListWidget()
        self.file_list_widget.currentItemChanged.connect(self.chk_current_item_changed) # listwidget 아이템 선택시 이벤트
        file_list_layout.addWidget(self.file_list_widget)

        # 파일리스트 버튼 위젯
        file_list_btn_layout.addWidget(self.video_file_list_btn)

        # play_video_btn layout widget
        play_video_btn_layout.addWidget(self.video_start_btn)  # 영상 재생 버튼
        play_video_btn_layout.addWidget(self.video_pause_btn)  # 영상 일시정지 버튼
        play_video_btn_layout.addWidget(self.video_stop_btn)  # 영상 초기화 버튼

        # 정보 레이아웃 전체
        self.information_qlabel = QLabel(self)
        self.information_qlabel.setFont(QFont('Arial', 15))
        self.information_qlabel.setText("Information")
        self.information_qlabel.setFixedSize(100, 20)

        self.information_select_qlabel = QLabel(self)
        self.information_select_qlabel.setFont(QFont('Arial', 13))
        self.information_select_qlabel.setText("Info Type")
        self.information_select_qlabel.setFixedSize(100, 20)

        information_line1 = QFrame()
        information_line1.setFrameShape(QFrame.HLine)
        information_line1.setFrameShadow(QFrame.Sunken)

        #  Radio button select info
        self.radio_box_plot = QGroupBox()
        self.radio_box_plot_layout = QHBoxLayout()
        self.radio_box_plot.setLayout(self.radio_box_plot_layout)
        self.radio_box_plot_layout.addWidget(self.information_select_qlabel)

        self.information_select_total = QRadioButton('Total')
        self.information_select_total.setChecked(True)
        self.information_select_total.clicked.connect(self.change_plot)
        self.radio_box_plot_layout.addWidget(self.information_select_total)

        self.information_select_object = QRadioButton('Object')
        self.information_select_object.clicked.connect(self.change_plot)
        self.radio_box_plot_layout.addWidget(self.information_select_object)

        self.information_select_action = QRadioButton('Action')
        self.information_select_action.clicked.connect(self.change_plot)
        self.radio_box_plot_layout.addWidget(self.information_select_action)
        self.radio_box_plot_layout.addWidget(self.reload_plot_data_btn)

        self.information_select_qlabel.setMaximumWidth(120)
        self.information_select_total.setMaximumWidth(120)
        self.information_select_object.setMaximumWidth(120)
        self.information_select_action.setMaximumWidth(120)
        self.reload_plot_data_btn.setMaximumWidth(120)

        information_label_layout.addWidget(self.information_qlabel, alignment=Qt.AlignLeft)
        information_label_layout.addWidget(self.radio_box_plot, alignment=Qt.AlignRight)

        all_information_layout.addLayout(information_label_layout)
        all_information_layout.addWidget(information_line1)

        # information_layout widget
        self.camera_object_groupbox_widget = draw_camera_object_groupbox() # 객체 통계 그룹박스 불러오기(QGroupBox 리턴)
        information_layout.addWidget(self.camera_object_groupbox_widget)

        self.camera_action_groupbox_widget = draw_camera_action_groupbox() # 액션 통계 그룹박스 불러오기(QGroupBox 리턴)
        information_layout.addWidget(self.camera_action_groupbox_widget)

        self.total_json_fp = f'logs/log_total.json'
        self.plot_date = []
        self.plot_action = []
        self.plot_object = []
        self.init_plot()  # 월별 통계 plot 불러오기
        information_layout.addWidget(self.plot_widget)

        # 레이아웃 추가
        video_qlabel_layout.addLayout(video_sync_layout)
        all_play_video_layout.addLayout(video_qlabel_layout)
        all_play_video_layout.addLayout(video_top_line_layout)
        all_play_video_layout.addLayout(play_video_label_layout)
        all_play_video_layout.addLayout(play_video_layout)
        all_play_video_layout.addLayout(play_video_btn_layout)

        all_file_list_layout.addLayout(file_list_layout)
        all_file_list_layout.addLayout(file_list_btn_layout)

        information_widget.setLayout(information_layout)
        all_information_layout.addLayout(information_layout)
        all_information_layout.addWidget(information_widget)

        main_layout.addLayout(all_play_video_layout)
        main_layout.addLayout(empty_box_layout)
        main_layout.addLayout(all_file_list_layout)
        main_layout.addLayout(empty_box_layout2)
        main_layout.addLayout(all_information_layout)

        self.setLayout(main_layout)

        # Sync 플래그 추가
        self.vis1_ready = False
        self.vis2_ready = False
        self.vis_terminate = False

        self.video_path = None
        self.init_count()

    def video_start(self): # 영상 재생 함수
        if self.file_list_widget.currentItem() == None: # listwidget 아이템 미선택시 바로 리턴
            return
        else:
            self.video_path = self.file_list_widget.currentItem().text()  # listwidget으로부터 선택된 영상 경로 불러오기
            if self.last_video != self.video_path: # 마지막 재생 영상과 현재 선택된 영상 미일치시 비디오 처음부터 다시 재생
                self.last_video = self.video_path
                if self.video_load == True:
                    self.video_stop()

            if self.video_load: # 영상 일시정지 상태에서 다시 재생시
                self.video_play = True
                self.model_init_log.setText("State : Start Video")
            else:               # 초기 영상 Loading
                self.video_load = True
                self.video_play = True
                self.vis_terminate = False  # Sync 반복문 정지 플래그 비활성화
                self.init_count()  # Count 0으로 초기화

                # 모델 사용 시 Queue에 계속 쌓이게 되어 메모리 이슈가 발생
                # 최대 사이즈를 slowfast에서 한꺼번에 나오는 64 프레임 을 기준으로 변경, 메모리 이슈에 의해 처리함
                self.frame_q = Queue(maxsize=64)  # 선택된 영상의 frame이 담길 Queue
                self.detect_q = Queue(maxsize=64) # 추론된 영상의 frame이 담길 Queue
                self.action_detect_q = Queue(maxsize=64) # 행동 추론 영상의 frame이 담길 Queue

                # object tracking에 사용되는 리스트 및 초기화 값
                self.track_twenty_four_frame_list = []  # 240 프레임, 10초(24fps 기준)동안 저장되는 ID 값 리스트
                self.track_count = 0    # 프레임 카운트
                self.prev_frame_class_list = [] # 전 프레임 클래스 저장 리스트

                self.action_stop_pipe_parent, self.action_stop_pipe_child = Pipe()
                self.object_model_init_parent_pipe, self.object_model_init_child_pipe = Pipe()
                self.action_model_init_parent_pipe, self.action_model_init_child_pipe = Pipe()

                self.frame_reader_p = Process(target=lv.read_frames, args=(self.frame_q, self.detect_q, self.video_path,
                                                                           self.object_model_init_child_pipe),
                                              name="READ_FRAME_P") # 쓰레드를 통한 영상 프레임 읽기(원본 영상 프레임은 frame_q에, 추론 영상 프레임은 detect_q에 쌓임)

                executor = ThreadPoolExecutor(1) # 쓰레드를 통한 원본 영상 및 추론 영상 프레임 가시화
                executor.submit(self.visual_process)

                # ADD 행동 프로세스
                self.frame_reader_p2 = Process(target=lv.slowfast_read_frames, args=(self.action_detect_q, self.action_stop_pipe_child,
                                                                                     self.video_path, self.action_model_init_child_pipe),
                                               name="SLOWFAST_FRAME_P")

                executor2 = ThreadPoolExecutor(1)  # 쓰레드를 통한 원본 영상 및 추론 영상 프레임 가시화
                executor2.submit(self.visual_process2)

                self.frame_reader_p.daemon = True
                self.frame_reader_p.start()

                self.frame_reader_p2.daemon = False
                self.frame_reader_p2.start()

                # UI 상태 표시 프로세스, FPS 상태 표시 프로세스
                self.model_init_log.setText("State : Model Initializing...")
                executor3 = ThreadPoolExecutor(1)
                executor3.submit(self.state_log)

    def video_pause(self): # 영상 일시정지 함수
        self.video_play = False
        # self.action_stop_pipe_parent.send("pause")
        self.model_init_log.setText("State : Pause Video")

    def video_stop(self): # 영상 초기화 함수
        # pipe 신호를 이용해서 프로세스 종료
        if self.frame_reader_p2.is_alive():
            self.action_stop_pipe_parent.send("stop")
            while True:
                if str(self.action_stop_pipe_parent.poll()):
                    if self.action_stop_pipe_parent.recv() == "done":
                        self.frame_reader_p2.terminate()
                        break

        self.frame_q = None
        self.detect_q = None
        self.action_detect_q = None
        self.video_load = False
        self.video_play = False
        if self.frame_reader_p.is_alive():
            self.frame_reader_p.terminate()

        # Sync 플래그 초기화
        self.vis_terminate = True  # Sync 반복문 정지 플래그 활성화
        self.vis1_ready = False
        self.vis2_ready = False

        self.model_init_log.setText("State : Stop Video")
        self.drop_log()

    # 그래프 드로잉 함수
    def draw_plot(self, x, y, width=0.4, brush=(180, 180, 180), pen=(100, 100, 100)):
        self.plot_widget.setXRange(-1, len(x), padding=0)  # 초기 X 축 범위 지정
        self.plot_widget.setYRange(0, max(y) + (max(y) * 0.01), padding=0)  # 초기 Y 축 범위 지정
        self.bargraph = pg.BarGraphItem(x=range(len(x)), height=y, width=width, brush=brush, pen=pen)  # barchart 생성
        self.plot_widget.addItem(self.bargraph)  # widget에 생성한 bargraph 추가
        month_labels = [(n, m) for n, m in zip(range(len(x)), x)]  # X축의 넘버와 날짜를 짝짓기
        ax = self.plot_widget.getAxis('bottom')  # widget의 아래 축을 선택
        ax.setTicks([month_labels])  # 선택한 축의 값을 [(기존 값, 대체할 값), ...] 형식으로 대체

    # 그래프 초기 설정 함수
    def init_plot(self):
        pg.setConfigOptions(background=(240, 240, 240), foreground=(0, 0, 0))  # pyqtgraph 옵션 설정(전경,배경 색 지정)
        self.plot_widget = pg.PlotWidget(title="Detect Count per Month")
        self.plot_widget.showGrid(x=False, y=False)  # x축, y축 격자 무늬 제거
        if os.path.isfile(self.total_json_fp):
            with open(self.total_json_fp, 'r') as total_json_file:  # JSON 읽기
                total_json_data = json.load(total_json_file)
                for date, data in sorted(total_json_data.items()):  # 정렬 후 리스트에 데이터 나누기
                    self.plot_date.append(date)
                    self.plot_action.append(data['total_action'])
                    self.plot_object.append(data['total_object'])
            if self.information_select_total.isChecked():  # 체크박스 체크 현황에 따라 다른 그래프 드로잉
                plot_total = [(a + b) for a, b in zip(self.plot_action, self.plot_object)]
                self.draw_plot(self.plot_date, plot_total)
            elif self.information_select_object.isChecked():
                self.draw_plot(self.plot_date, self.plot_object)
            elif self.information_select_action.isChecked():
                self.draw_plot(self.plot_date, self.plot_action)
            else:
                pass
        else:
            self.draw_plot(['No Total Data. Press Refresh.'], [999])

    def reload_plot_data(self):  # drop_log로 생성된 로그들을 모두 읽어 그래프를 그리기 위한 JSON 하나로 함축
        total_dict = defaultdict(dict)
        excluded = ['log_total.json']  # 이미 모아져 있는 파일 무시
        for (root, dirs, files) in os.walk('logs'):  # 로그들의 데이터를 월별 dict로 카운트
            for file in files:
                if file not in excluded:
                    date = root.split('\\')[-1]
                    total_dict[f'{date}'] = {
                        'total_action': 0,
                        'total_object': 0,
                        'person': 0,
                        'car': 0,
                        'boat': 0,
                        'sos': 0,
                        'fall_down': 0
                    }
                    with open(f'{root}/{file}') as json_file:
                        json_data = json.load(json_file)
                        for v in json_data.values():
                            for key, value in v.items():
                                total_dict[f'd{date}'][key] += value
        if total_dict:
            with open(self.total_json_fp, 'w') as total_json_file:  # total_dict를 기반으로 파일 새로 작성
                total_json_data = OrderedDict(total_dict)
                json.dump(total_json_data, total_json_file, ensure_ascii=False, indent='\t')
            self.plot_widget.removeItem(self.bargraph)  # 기존 그래프 삭제
            self.init_plot()  # 그래프 widget 자체를 새로 고침

    def change_plot(self):  # 체크박스에 따라 다른 그래프 표시
        if self.information_select_total.isChecked():
            plot_total = [(a + b) for a, b in zip(self.plot_action, self.plot_object)]
            self.plot_widget.removeItem(self.bargraph)  # widget에 있는 기존 bargraph 삭제
            self.draw_plot(self.plot_date, plot_total)
        elif self.information_select_object.isChecked():
            self.plot_widget.removeItem(self.bargraph)
            self.draw_plot(self.plot_date, self.plot_object)
        elif self.information_select_action.isChecked():
            self.plot_widget.removeItem(self.bargraph)
            self.draw_plot(self.plot_date, self.plot_action)
        else:
            pass

    def init_count(self):  # 영상 정지 후 시작시에 Count를 0으로 초기화 및 변수 할당
        # 클래스 정의
        self.action_cls = ['sos', 'fall_down', 'total_action']
        self.object_cls = ['car', 'person', 'boat', 'total_object']
        self.total_object_count = self.camera_object_groupbox_widget.findChild(QLabel, 'total_object_count')
        self.person_count = self.camera_object_groupbox_widget.findChild(QLabel, 'person_count')
        self.car_count = self.camera_object_groupbox_widget.findChild(QLabel, 'car_count')
        self.boat_count = self.camera_object_groupbox_widget.findChild(QLabel, 'boat_count')
        self.total_action_count = self.camera_action_groupbox_widget.findChild(QLabel, 'total_action_count')
        self.sos_count = self.camera_action_groupbox_widget.findChild(QLabel, 'sos_count')
        self.fall_down_count = self.camera_action_groupbox_widget.findChild(QLabel, 'fall_down_count')
        self.total_object_count.setText('0')
        self.person_count.setText('0')
        self.car_count.setText('0')
        self.boat_count.setText('0')
        self.total_action_count.setText('0')
        self.sos_count.setText('0')
        self.fall_down_count.setText('0')

    def cls_count(self, detect_count_dict: dict):  # Object Count 갱신 함수
        # detect_count_dict에 포함된 Object 개수 더하기
        # total_obj_cnt = int(self.total_object_count.text())
        total_act_cnt = int(self.total_action_count.text())
        for cls, value in detect_count_dict.items():
            if cls in self.action_cls + self.object_cls:
                cnt = int(eval(f'self.{cls}_count.text()'))
                # if cls in self.object_cls:
                #     cnt += value
                #     total_obj_cnt += value
                if cls in self.action_cls:
                    cnt += 1
                    total_act_cnt += 1
                eval(f"self.{cls}_count.setText(f'{{cnt}}')")
        # self.total_object_count.setText(f'{total_obj_cnt}')
        self.total_action_count.setText(f'{total_act_cnt}')

    def drop_log(self):  # 로그 JSON 드롭
        # ../2023.01.16/video_name
        # video_date_name = self.video_path.split('\\')[-1]
        video_name = self.video_path.split('/')[-1]
        print("video_name : ", video_name)

        # date
        folder_name = self.video_path.split('/')[-2]
        # video_name = video_date_name.split('/')[-1]
        log_json_fp = f'logs/{folder_name[:-3]}/{folder_name}.json'
        os.makedirs(os.path.split(log_json_fp)[0], exist_ok=True)
        if os.path.isfile(log_json_fp):
            with open(log_json_fp, 'r') as log_json_file:
                log_json_data = OrderedDict(json.load(log_json_file))
        else:
            log_json_data = OrderedDict()
        log_json_data[f'{video_name}'] = {
            'total_action': int(self.total_action_count.text()),
            'total_object': int(self.total_object_count.text()),
            'person': int(self.person_count.text()),
            'car': int(self.car_count.text()),
            'boat': int(self.boat_count.text()),
            'sos': int(self.sos_count.text()),
            'fall_down': int(self.fall_down_count.text())
        }
        with open(log_json_fp, 'w') as drop_json:
            json.dump(log_json_data, drop_json, ensure_ascii=False, indent='\t')

    def yolo_object_tracking_count(self, id_label_count_list):
        total_obj_cnt = int(self.total_object_count.text())

        for i, item in enumerate(id_label_count_list):
            if self.track_count == 0:
                id = item[0]
                label = item[1]
                if label in self.object_cls:
                    cnt = int(eval(f'self.{label}_count.text()'))
                    cnt += 1
                    total_obj_cnt += 1
                    eval(f"self.{label}_count.setText(f'{{cnt}}')")
                if item not in self.track_twenty_four_frame_list:
                    self.track_twenty_four_frame_list.append(item)
                if item not in self.prev_frame_class_list:
                    self.prev_frame_class_list.append(item)
            else:
                # 전 프레임 클래스 리스트 안에 있을 경우(True), 240프레임 리스트 안에 있을 경우(True)
                if item in self.prev_frame_class_list and item in self.track_twenty_four_frame_list:
                    continue
                # 전 프레임 클래스 리스트 안에 없을 경우(False), 240프레임 리스트 안에 있을 경우(True)
                elif item not in self.prev_frame_class_list and item in self.track_twenty_four_frame_list:
                    continue
                # 전 프레임 클래스 리스트 안에 있을 경우(True), 240프레임 리스트 안에 없을 경우(False)
                elif item in self.prev_frame_class_list and item not in self.track_twenty_four_frame_list:
                    self.track_twenty_four_frame_list.append(item)
                    continue
                else:
                    id = item[0]
                    label = item[1]
                    if label in self.object_cls:
                        cnt = int(eval(f'self.{label}_count.text()'))
                        cnt += 1
                        total_obj_cnt += 1
                        eval(f"self.{label}_count.setText(f'{{cnt}}')")
                        self.track_twenty_four_frame_list.append(item)

        self.track_count += 1
        # 240 프레임 동안 저장된 id 초기화(0~239)
        if self.track_count == 239:
            self.track_count = 0
            self.track_twenty_four_frame_list.clear()
        self.prev_frame_class_list = id_label_count_list
        self.total_object_count.setText(f'{total_obj_cnt}')


    def visual_process(self): # 영상 가시화 함수
        self.obj_prev_time = time.time()
        self.obj_fps_widget_label.clear()
        self.fps_widget_label.clear()
        while True:
            self.vis1_ready = False
            if self.frame_q.qsize() > 0 and self.video_play: # 영상이 재생 중이며 frame_q에 frame이 하나 이상 존재할 때 가시화
                frame = self.frame_q.get()
                frame = self.convert_cv_qt(frame)
                detect_result = self.detect_q.get()
                detect_frame = self.convert_cv_qt(detect_result[0])  # detect_q 중 frame
                self.vis1_ready = True

                while not self.vis2_ready and self.video_sync:
                    if self.vis1_ready and self.vis2_ready or self.vis_terminate:
                        break

                self.original_video.setImage(frame)
                self.detected_video.setImage(detect_frame)

                cur_time = time.time()
                fps = 1 / (cur_time - self.obj_prev_time)
                self.obj_prev_time = cur_time
                self.obj_fps_str = "%0.1f" % fps

                if self.video_sync:
                    self.fps_widget_label.setText("SYNC_FPS : " + self.obj_fps_str)
                else:
                    self.obj_fps_widget_label.setText("OBJ_FPS : " + self.obj_fps_str)

                # object tracking에 맞게 클래스 카운트 변경
                id_label_count_list = detect_result[1]
                if not self.vis_terminate:
                    self.yolo_object_tracking_count(id_label_count_list=id_label_count_list)
                time.sleep(0.015)
            else:
                continue

    # ADD action recognize visualization
    def visual_process2(self): # 영상 가시화 함수
        self.act_prev_time = time.time()
        self.act_fps_widget_label.clear()
        while True:
            self.vis2_ready = False
            if self.action_detect_q.qsize() > 0 and self.video_play:
                self.recognize_frame = self.action_detect_q.get()
                if type(self.recognize_frame) == dict:
                    ## 행동 confidence 값
                    action_count = self.recognize_frame
                    if not self.vis_terminate:
                        self.cls_count(action_count)
                else:
                    self.vis2_ready = True

                    while not self.vis1_ready and self.video_sync:
                        if self.vis1_ready and self.vis2_ready or self.vis_terminate:
                            break
                    self.recognize_frame = self.convert_cv_qt(self.recognize_frame)
                    self.recognize_video.setImage(self.recognize_frame)

                    cur_time = time.time()
                    fps = 1 / (cur_time - self.act_prev_time)
                    self.act_prev_time = cur_time
                    self.act_fps_str = "%0.1f" % fps

                    if not self.video_sync:
                        self.act_fps_widget_label.setText("ACT_FPS : " + self.act_fps_str)

                    time.sleep(0.015)
            else:
                continue

    def convert_cv_qt(self, frame): # cv2 이미지를 QImage 형태로 변환하는 함수(640*360 사이즈로 자동 변환)
        rgb_image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        h, w, c = rgb_image.shape
        bytes_per_line = c * w
        convert_to_Qt_format = QImage(rgb_image.data, w, h, bytes_per_line, QImage.Format_RGB888)
        scaled_img = convert_to_Qt_format.scaled(620, 360, Qt.IgnoreAspectRatio)
        return scaled_img

    def chk_current_item_changed(self): # listwidget 아이템 선택시 발생하는 이벤트
        print("Clicked Video : " + self.file_list_widget.currentItem().text())

    def video_file_list_path(self):
        file_list_path = QFileDialog.getExistingDirectory(self)
        for root, dirs, files in os.walk(file_list_path):
            for file in files:
                _, extension = os.path.splitext(file)
                if extension == ".mp4":
                    self.file_list_widget.addItem(root + "/" + file)

    def video_sync_clicked(self):
        if self.btn_video_sync_true.isChecked():
            self.video_sync = True
        elif self.btn_video_sync_false.isChecked():
            self.video_sync = False
        else:
            pass

    def state_log(self):
        object_model_init_flag = False
        action_model_init_flag = False
        while True:
            # UI 상태 표시
            if str(self.object_model_init_parent_pipe.poll()):
                if self.object_model_init_parent_pipe.recv() == "model_init_done":
                    self.model_init_log.setText("State : Object Model Init... Done")
                    object_model_init_flag = True
            if str(self.action_model_init_parent_pipe.poll()):
                if self.action_model_init_parent_pipe.recv() == "model_init_done":
                    self.model_init_log.setText("State : Action Model Init... Done")
                    action_model_init_flag = True
            if object_model_init_flag and action_model_init_flag:
                self.model_init_log.setText("State : All Model Init... Done, Start Video")
                break
