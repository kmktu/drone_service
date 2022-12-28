import torch
from pathlib import Path
import sys, os
import numpy as np
import threading
from queue import Queue
from collections import defaultdict
sys.path.append("..")

FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # YOLOv5 root directory
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative


from utils.torch_utils import select_device
from models.common import DetectMultiBackend
from utils.augmentations import letterbox
from utils.plots import Annotator, colors, save_one_box
from utils.general import (LOGGER, check_img_size, check_requirements, colorstr,
                           non_max_suppression, scale_coords, strip_optimizer, xyxy2xywh)

class ObjectDetection():
    def __init__(self):
        self.model_path = 'yolov5n.pt'
        self.device = '0'
        self.imgsz = (640, 640)
        self.conf_thres = 0.25
        self.iou_thres = 0.45
        self.classes = None
        self.agnostic_nms = True
        self.max_det = 1000
        self.save_crop = False
        self.view_img = False,  # show results
        self.save_txt = False,  # save results to *.txt
        self.save_conf = False,  # save confidences in --save-txt labels
        self.save_crop = False,  # save cropped prediction boxes
        self.save_img = False,
        self.agnostic_nms = True,  # class-agnostic NMS
        self.update = False,  # update all models
        self.line_thickness = 3,  # bounding box thickness (pixels)
        self.hide_labels = False,  # hide labels
        self.hide_conf = False,  # hide confidences
        self.half = False,  # use FP16 half-precision inference
        self.dnn = False,  # use OpenCV DNN for ONNX inference
        self.queue = Queue()
        self.model_init()

    @torch.no_grad()
    def model_init(self):
        self.device = select_device(self.device)
        self.model = DetectMultiBackend(self.model_path, device=self.device, dnn=False, fp16=False)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup()

    @torch.no_grad()
    def inference_img(self, img_to_infer):
        img = letterbox(img_to_infer, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0, 1))[::-1]
        im = np.ascontiguousarray(img)

        seen, windows, dt = 0, [], [0.0, 0.0, 0.0]

        im = torch.from_numpy(im)
        im = im.to(self.device)
        im = im.half() if self.model.fp16 else im.float()
        im /= 255
        if len(im.shape) == 3:
            im = im[None]

        self.pred = self.model(im)

        torch.cuda.synchronize()
        self.pred = non_max_suppression(self.pred, self.conf_thres, self.iou_thres, self.classes,
                                   self.agnostic_nms, max_det=self.max_det)

        for i, det in enumerate(self.pred):
            seen += 1

            self.im0 = img_to_infer.copy()
            self.label_count = defaultdict(int)  # 라벨 카운트 딕셔너리 생성

            # gn = torch.tensor(self.im0.shape)[[1, 0, 1, 0]]

            # imc = self.im0.copy() if self.save_crop else img_to_infer

            self.annotator = Annotator(self.im0, line_width=self.line_thickness, example=str(self.names))
            # LOGGER.info(f"{'Class'. ljust(15)} {'Confidence'.ljust(20)} {'BBOX'.ljust(30)}")

            if len(det):
                det[:, :4] = scale_coords(im.shape[2:], det[:, :4], self.im0.shape).round()

                for *xyxy, conf, cls in reversed(det):
                    c = int(cls.item())
                    bbox = list(int(v.item()) for v in xyxy)
                    self.label_count[f'{self.names[c]}'] += 1  # 클래스 카운트
                    # LOGGER.info(f"{colorstr('bold', self.names[c].ljust(15))}"
                    #             f"{colorstr('bold', str(conf.item()).ljust(20))}"
                    #             f"{colorstr('bold', str(bbox).ljust(30))}")

                    # label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{conf:.2f}')
                    # if self.hide_labels:
                    #     print("label : ", self.names[c], self.hide_labels)
                    label = None if self.hide_labels else (self.names[c] if self.hide_conf else f'{self.names[c]} {conf:.2f}')

                    self.annotator.box_label(xyxy, label, color=colors(c, True))

            self.im0 = self.annotator.result()
        self.queue.put((self.im0, self.label_count))  # 이미지와 클래스 개수를 tuple로 묶어 put

    # @pyqtSlot(np.ndarray)
    def thread_job(self, frame):
        while True:
            t = threading.Thread(target=self.inference_img, args=(frame))
            t.start()

    def get_data(self):
        return self.queue.get()
