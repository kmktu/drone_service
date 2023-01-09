import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import numpy as np
from pathlib import Path
import torch
from queue import Queue

sys.path.append("..")
FILE = Path(__file__).resolve()
ROOT = FILE.parents[0]  # yolov5 strongsort root directory
WEIGHTS = ROOT / 'weights'

if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))  # add ROOT to PATH
if str(ROOT / 'yolov5') not in sys.path:
    sys.path.append(str(ROOT / 'yolov5'))  # add yolov5 ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort'))  # add strong_sort ROOT to PATH
if str(ROOT / 'trackers' / 'strong_sort' / 'configs') not in sys.path:
    sys.path.append(str(ROOT / 'trackers' / 'strong_sort' / 'configs'))

ROOT = Path(os.path.relpath(ROOT, Path.cwd()))  # relative

from yolov5.models.common import DetectMultiBackend
from yolov5.utils.general import (LOGGER, Profile, non_max_suppression, scale_boxes, strip_optimizer)
from yolov5.utils.torch_utils import select_device
from yolov5.utils.plots import Annotator, colors
from utils.segment.general import masks2segments, process_mask, process_mask_native
from utils.augmentations import letterbox
from trackers.multi_tracker_zoo import create_tracker

class ObjectDetection():
    def __init__(self):
        self.source = '0'
        self.yolo_weights = WEIGHTS / 'yolov5m.pt'  # model.pt path(s),
        self.reid_weights = WEIGHTS / 'osnet_x0_25_msmt17.pt'  # model.pt path,
        self.tracking_method = 'strongsort'
        self.imgsz = (640, 640)  # inference size (height, width)
        self.conf_thres = 0.25  # confidence threshold
        self.iou_thres = 0.45  # NMS IOU threshold
        self.max_det = 1000  # maximum detections per image
        self.device = ''  # cuda device, i.e. 0 or 0,1,2,3 or cpu
        self.show_vid = True  # show results
        self.save_txt = False  # save results to *.txt
        self.save_conf = False  # save confidences in --save-txt labels
        self.save_crop = False  # save cropped prediction boxes
        self.save_trajectories = False # save trajectories for each track
        self.save_vid = False  # save confidences in --save-txt labels
        self.nosave = False  # do not save images/videos
        self.classes = None  # filter by class: --class 0, or --class 0 2 3
        self.agnostic_nms = False  # class-agnostic NMS
        self.augment = False  # augmented inference
        self.visualize = False  # visualize features
        self.update = False  # update all models
        self.project = ROOT / 'runs/track'  # save results to project/name
        self.name = 'exp'  # save results to project/name
        self.exist_ok = False  # existing project/name ok, do not increment
        self.line_thickness = 2  # bounding box thickness (pixels)
        self.hide_labels = False  # hide labels
        self.hide_conf = False  # hide confidences
        self.hide_class = False  # hide IDs
        self.half = False  # use FP16 half-precision inference
        self.dnn = False  # use OpenCV DNN for ONNX inference
        self.vid_stride = 1  # video frame-rate stride
        self.retina_masks = False
        self.queue = Queue()

        self.model_init()

    @torch.no_grad()
    def model_init(self):
        # Load model
        self.device = select_device(self.device)
        self.is_seg = '-seg' in str(self.yolo_weights)
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup()

        nr_sources = 1

        # Create as many strong sort instances as there are video sources
        self.tracker_list = []
        for i in range(nr_sources):
            self.tracker = create_tracker(self.tracking_method, self.reid_weights, self.device, self.half)
            self.tracker_list.append(self.tracker, )
            if hasattr(self.tracker_list[i], 'model'):
                if hasattr(self.tracker_list[i].model, 'warmup'):
                    self.tracker_list[i].model.warmup()
        self.outputs = [None] * nr_sources

        self.seen, self.windows, self.dt = 0, [], (Profile(), Profile(), Profile(), Profile())
        self.curr_frames, self.prev_frames = [None] * nr_sources, [None] * nr_sources


    @torch.no_grad()
    def inference_tracking(self, img_to_infer):
        img = letterbox(img_to_infer, self.imgsz, stride=self.stride, auto=self.pt)[0]
        img = img.transpose((2, 0 , 1))[::-1]
        im = np.ascontiguousarray(img)

        with self.dt[0]:
            im = torch.from_numpy(im).to(self.device)
            im = im.half() if self.half else im.float()
            im /= 255.0
            if len(im.shape) == 3:
                im = im[None]
                
        with self.dt[1]:
            if self.is_seg:
                pred, proto = self.model(im)[:2] # augment, visualize 삭제
            else:
                pred = self.model(im)

        with self.dt[2]:
            if self.is_seg:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det, nm=32)
            else:
                pred = non_max_suppression(pred, self.conf_thres, self.iou_thres, self.classes, self.agnostic_nms,
                                           max_det=self.max_det)
        for i, det in enumerate(pred):
            self.seen += 1

            # 웹캠 사용 여부, 파일 경로, txt 파일 저장 부분 삭제
            im0 = img_to_infer.copy()

            # Tracking 하기 위해 프레임 저장
            self.curr_frames[i] = im0

            # annotator
            self.annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            # ID, 라벨 카운트 리스트
            self.id_label_count_list = []

            # tracker 확인 및 프레임 업데이트
            if hasattr(self.tracker_list[i], 'tracker') and hasattr(self.tracker_list[i].tracker, 'camera_updates'):
                if self.prev_frames[i] is not None and self.curr_frames[i] is not None:
                    self.tracker_list[i].tracker.camera_update(self.prev_frames[i], self.curr_frames[i])

            if det is not None and len(det):
                if self.is_seg:
                    if self.retina_masks:
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0.shape).round()  # rescale boxes to im0 size
                        masks = process_mask_native(proto[i], det[:, 6:], det[:, :4], im0.shape[:2])  # HWC
                    else:
                        masks = process_mask(proto[i], det[:, 6:], det[:, :4], im.shape[2:], upsample=True)  # HWC
                        det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                                 im0.shape).round()  # rescale boxes to im0 size
                else:
                    det[:, :4] = scale_boxes(im.shape[2:], det[:, :4],
                                             im0.shape).round()  # rescale boxes to im0 size
                # pass detections to strongsort
                with self.dt[3]:
                    self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)

                if len(self.outputs[i]) > 0:
                    for j, (output) in enumerate(self.outputs[i]):

                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]

                        # 박스 그리는 부분
                        if self.save_vid or self.save_crop or self.show_vid:  # Add bbox to image
                            c = int(cls)  # integer class
                            id = int(id)  # integer id
                            # 0:person, 2:car, 8:boat in coco
                            if self.names[c] == "person" or self.names[c] == "car" or self.names[c] == "boat":
                                label = None if self.hide_labels else (f'{id} {self.names[c]}' if self.hide_conf else
                                                                       (f'{id} {conf:.2f}' if self.hide_class else
                                                                        f'{id} {self.names[c]} {conf:.2f}'))

                                color = colors(c, True)
                                self.annotator.box_label(bbox, label, color=color)
                                if self.is_seg:
                                    # Mask plotting
                                    self.annotator.masks(
                                        masks,
                                        colors=[colors(x, True) for x in det[:, 5]],
                                        im_gpu=torch.as_tensor(im0, dtype=torch.float16).to(self.device).permute(2, 0,
                                                                                                                 1).flip(
                                            0).contiguous() / 255 if self.retina_masks else im[i]
                                    )
                                if self.save_trajectories and self.tracking_method == 'strongsort':
                                    q = output[7]
                                    self.tracker_list[i].trajectory(im0, q, color=color)

                                self.id_label_count_list.append([id, self.names[c]])
                            else:
                                continue
            else:
                pass

            im0 = self.annotator.result()
            self.queue.put((im0, self.id_label_count_list))

            self.prev_frames[i] = self.curr_frames[i]

            # Print total time (preprocessing + inference + NMS + tracking)
        # LOGGER.info(
        #     f"{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in self.dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")
        #
        # # Print results
        # t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        # LOGGER.info(
        #     f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {self.tracking_method} update per image at shape {(1, 3, *self.imgsz)}' % t)

        if self.update:
            strip_optimizer(self.yolo_weights)  # update model (to fix SourceChangeWarning)

    def get_data(self):
        return self.queue.get()
