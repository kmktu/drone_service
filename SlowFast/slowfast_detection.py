import os, sys
from pathlib import Path
import cv2

sys.path.append("..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from slowfast.config.defaults import assert_and_infer_cfg
from slowfast.utils.misc import launch_job
from slowfast.visualization.video_visualizer import VideoVisualizer
from slowfast.visualization.demo_loader import VideoManager
from slowfast.visualization.async_predictor import AsyncDemo, AsyncVis
from slowfast.visualization.predictor import ActionPredictor
from slowfast.utils import logging
from slowfast.utils.parser import load_config, parse_args
from slowfast.visualization.utils import TaskInfo
from SlowFast.slowfast.utils.misc import get_class_names

import tqdm
import numpy as np
import torch
import time

# from queue import Queue
# from multiprocessing import Queue
from torch.multiprocessing import Queue

logger = logging.get_logger(__name__)

class SlowFastDetection():
    def __init__(self):
        self.model_init()

    def model_init(self):
        self.cfg_path = "configs/AVA/SLOWFAST_32x2_R50_SHORT5.yaml"
        self.num_task = 0

        self.args = parse_args()
        for path_to_config in self.args.cfg_files:
            self.cfg = load_config(self.args, path_to_config)
            self.cfg = assert_and_infer_cfg(self.cfg)
            # print("AAA : ", self.cfg)

        np.random.seed(self.cfg.RNG_SEED)
        torch.manual_seed(self.cfg.RNG_SEED)
        # Setup logging format.
        logging.setup_logging(self.cfg.OUTPUT_DIR)
        logger.info("Run demo with config")
        # logger.info(self.cfg)

        common_classes = (
            self.cfg.DEMO.COMMON_CLASS_NAMES
            if len(self.cfg.DEMO.LABEL_FILE_PATH) != 0
            else None
        )

        video_vis = VideoVisualizer(
            num_classes=self.cfg.MODEL.NUM_CLASSES,
            class_names_path=self.cfg.DEMO.LABEL_FILE_PATH,
            top_k=self.cfg.TENSORBOARD.MODEL_VIS.TOPK_PREDS,
            thres=self.cfg.DEMO.COMMON_CLASS_THRES,
            lower_thres=self.cfg.DEMO.UNCOMMON_CLASS_THRES,
            common_class_names=common_classes,
            colormap=self.cfg.TENSORBOARD.MODEL_VIS.COLORMAP,
            mode=self.cfg.DEMO.VIS_MODE,
        )

        self.async_vis = AsyncVis(video_vis, n_workers=self.cfg.DEMO.NUM_VIS_INSTANCES)

        if self.cfg.NUM_GPUS <= 1:
            self.model = ActionPredictor(cfg=self.cfg, async_vis=self.async_vis)
        else:
            self.model = AsyncDemo(cfg=self.cfg, async_vis=self.async_vis)

        seq_len = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE

        assert (
                self.cfg.DEMO.BUFFER_SIZE <= seq_len // 2
        ), "Buffer size cannot be greater than half of sequence length."

        self.queue = Queue()
        print("slowfast model init")

    def video_path_input(self, input_cv_cap, video_path):
        self.input_cv_cap = input_cv_cap
        self.cfg.DEMO.INPUT_VIDEO = video_path

        self.source = (
            self.cfg.DEMO.WEBCAM if self.cfg.DEMO.WEBCAM > -1 else self.cfg.DEMO.INPUT_VIDEO
        )
        # self.frame_provider = VideoManager(self.cfg, self.source)
        self.frame_provider = self.frame_provider_init()

    def frame_provider_init(self):
        assert (
                self.cfg.DEMO.WEBCAM > -1 or self.cfg.DEMO.INPUT_VIDEO != ""
        ), "Must specify a data source as input."

        self.display_width = self.cfg.DEMO.DISPLAY_WIDTH
        self.display_height = self.cfg.DEMO.DISPLAY_HEIGHT

        if self.display_width > 0 and self.display_height > 0:
            self.input_cv_cap.set(cv2.CAP_PROP_FRAME_WIDTH, self.display_width)
            self.input_cv_cap.set(cv2.CAP_PROP_FRAME_HEIGHT, self.display_height)
        else:
            self.display_width = int(self.input_cv_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            self.display_height = int(self.input_cv_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))

        self.output_file = None
        if self.cfg.DEMO.OUTPUT_FPS == -1:
            self.output_fps = self.input_cv_cap.get(cv2.CAP_PROP_FPS)
        else:
            self.output_fps = self.cfg.DEMO.OUTPUT_FPS
        if self.cfg.DEMO.OUTPUT_FILE != "":
            self.output_file = self.get_output_file(
                self.cfg.DEMO.OUTPUT_FILE, fps=self.output_fps
            )

        self.id = -1
        self.buffer = []
        self.buffer_size = self.cfg.DEMO.BUFFER_SIZE
        self.seq_length = self.cfg.DATA.NUM_FRAMES * self.cfg.DATA.SAMPLING_RATE
        self.test_crop_size = self.cfg.DATA.TEST_CROP_SIZE
        self.clip_vis_size = self.cfg.DEMO.CLIP_VIS_SIZE

    def run_model(self, frames_list):
        self.id += 1
        task = TaskInfo()
        task.img_height = self.display_height
        task.img_width = self.display_width
        task.crop_size = self.test_crop_size
        task.clip_vis_size = self.clip_vis_size
        task.add_frames(self.id, frames_list)
        task.num_buffer_frames = 0 if self.id == 0 else self.buffer_size

        self.was_read = True

        if task is None:
            time.sleep(0.02)
        self.num_task += 1
        self.model.put(task)

        get_flag = False

        try:
            task = self.model.get()
            self.num_task -= 1
            yield task
        except IndexError:
            pass

        while self.num_task != 0:
            try:
                task = self.model.get()
                get_flag = True
                self.num_task -= 1
                yield task
                if get_flag:
                    break
            except IndexError:
                continue


    def get_data(self):
        return self.queue.get()

    def get_output_file(self, path, fps=30):
        """
        Return a video writer object.
        Args:
            path (str): path to the output video file.
            fps (int or float): frames per second.
        """
        return cv2.VideoWriter(
            filename=path,
            fourcc=cv2.VideoWriter_fourcc(*"mp4v"),
            fps=float(fps),
            frameSize=(self.display_width, self.display_height),
            isColor=True,
        )

    def slowfast_read_frames(self, action_detect_q, video_path):
        reader = cv2.VideoCapture(video_path)
        nframes = int(reader.get(cv2.CAP_PROP_FRAME_COUNT))

        self.video_path_input(reader, video_path)
        class_names_path = self.cfg.DEMO.LABEL_FILE_PATH
        # common_class_thres = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_THRES
        # uncommon_class_thres = inference_model_slowfast.cfg.DEMO.UNCOMMON_CLASS_THRES
        # common_class_names = inference_model_slowfast.cfg.DEMO.COMMON_CLASS_NAMES

        class_names, _, _ = get_class_names(class_names_path, None, None)
        class_names_dict = {}  # 클래스 이름별 탐지 결과 저장

        frames_list = []

        for ii in range(nframes):  # 영상의 마지막 프레임까지 반복
            _, frame = reader.read()
            frames_list.append(frame)
            if len(frames_list) == self.seq_length:
                for task in self.run_model(frames_list):
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
