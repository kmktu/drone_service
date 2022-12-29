import time
import cv2
from yolov5 import yolo_detection
from SlowFast.slowfast_detection import SlowFastDetection
from SlowFast.slowfast.utils.misc import get_class_names

class LoadVideo_model():
    def __init__(self):
        super(LoadVideo_model, self).__init__()
        self.model_init()

    def model_init(self):
        self.inference_model_yolo = yolo_detection.ObjectDetection()
        self.inference_model_slowfast = SlowFastDetection()
        print("model init")

    def read_frames(self, frame_q, detect_q, video_path):
        self.reader = cv2.VideoCapture(video_path)
        self.nframes = int(self.reader.get(cv2.CAP_PROP_FRAME_COUNT))

        for ii in range(self.nframes):
            while frame_q.qsize() > 50:
                time.sleep(1)
            _, frame = self.reader.read()

            frame_q.put(frame)
            self.inference_model_yolo.inference_img(frame)
            inference_img = self.inference_model_yolo.get_data()
            detect_q.put(inference_img)

    def slowfast_read_frames(self, action_detect_q, video_path):
        self.inference_model_slowfast.video_path_input(self.reader, video_path)
        class_names_path = self.inference_model_slowfast.cfg.DEMO.LABEL_FILE_PATH
        # common_class_thres = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_THRES
        # uncommon_class_thres = inference_model_slowfast.cfg.DEMO.UNCOMMON_CLASS_THRES
        # common_class_names = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_NAMES

        class_names, _, _ = get_class_names(class_names_path, None, None)
        class_names_dict = {}  # 클래스 이름별 탐지 결과 저장

        frames_list = []

        for ii in range(self.nframes):  # 영상의 마지막 프레임까지 반복
            _, frame = self.reader.read()

            frames_list.append(frame)
            if len(frames_list) == self.inference_model_slowfast.seq_length:
                for task in self.inference_model_slowfast.run_model(frames_list):

                    action_confidence_list = task.action_preds.tolist()
                    # action_confidence_list length => 12 => index start 0
                    # class_names => length => 13 => index start 1, first index value is None
                    # 클래스별 컨피던스 값
                    for i, class_name in enumerate(class_names):
                        if class_name != None:
                            class_names_dict[class_name] = action_confidence_list[0][i - 1]

                    for frame in task.frames[task.num_buffer_frames:]:
                        action_detect_q.put(frame)
                    action_detect_q.put(class_names_dict)
                frames_list.clear()


# def read_frames(frame_q, detect_q, video_path): # 영상의 프레임을 읽어와서 모델 추론을 수행하는 함수
#     inference_model_yolo = yolo_detection.ObjectDetection()
#     reader = cv2.VideoCapture(video_path)
#
#     nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     for ii in range(nframes):   # 영상의 마지막 프레임까지 반복
#         while frame_q.qsize() > 64: # frame_q에 50개 이상 frame이 쌓일 경우 sleep
#             time.sleep(1)
#         _, frame = reader.read()
#
#         frame_q.put(frame)
#         # 모델 추론부
#         inference_model_yolo.inference_img(frame)
#         inference_img = inference_model_yolo.get_data()
#         detect_q.put(inference_img)
#
# def slowfast_read_frames(frame_q, action_detect_q, video_path):
#     reader = cv2.VideoCapture(video_path)
#
#     nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))
#
#     inference_model_slowfast = SlowFastDetection()
#     inference_model_slowfast.video_path_input(reader, video_path)
#     class_names_path = inference_model_slowfast.cfg.DEMO.LABEL_FILE_PATH
#     # common_class_thres = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_THRES
#     # uncommon_class_thres = inference_model_slowfast.cfg.DEMO.UNCOMMON_CLASS_THRES
#     # common_class_names = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_NAMES
#
#     class_names, _, _ = get_class_names(class_names_path, None, None)
#     class_names_dict = {}   # 클래스 이름별 탐지 결과 저장
#
#     frames_list = []
#
#     for ii in range(nframes):  # 영상의 마지막 프레임까지 반복
#         _, frame = reader.read()
#
#         frames_list.append(frame)
#         if len(frames_list) == inference_model_slowfast.seq_length:
#             for task in inference_model_slowfast.run_model(frames_list):
#
#                 action_confidence_list = task.action_preds.tolist()
#                 # action_confidence_list length => 12 => index start 0
#                 # class_names => length => 13 => index start 1, first index value is None
#                 # 클래스별 컨피던스 값
#                 for i, class_name in enumerate(class_names):
#                     if class_name != None:
#                         class_names_dict[class_name] = action_confidence_list[0][i-1]
#
#                 for frame in task.frames[task.num_buffer_frames :]:
#                     action_detect_q.put(frame)
#                 action_detect_q.put(class_names_dict)
#             frames_list.clear()

