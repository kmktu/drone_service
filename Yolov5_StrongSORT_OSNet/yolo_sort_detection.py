import argparse
from collections import defaultdict

import os
# limit the number of cpus used by high performance libraries
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import sys
import platform
import numpy as np
from pathlib import Path
import torch
import torch.backends.cudnn as cudnn
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

import logging
from yolov5.models.common import DetectMultiBackend
from yolov5.utils.dataloaders import VID_FORMATS, LoadImages, LoadStreams
from yolov5.utils.general import (LOGGER, Profile, check_img_size, non_max_suppression, scale_boxes, check_requirements, cv2,
                                  check_imshow, xyxy2xywh, increment_path, strip_optimizer, colorstr, print_args, check_file)
from yolov5.utils.torch_utils import select_device, time_sync
from yolov5.utils.plots import Annotator, colors, save_one_box
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
        # show_vid = True
        # source = str(source)
        # save_img = not nosave and not source.endswith('.txt')  # save inference images
        # is_file = Path(source).suffix[1:] in (VID_FORMATS)
        # is_url = source.lower().startswith(('rtsp://', 'rtmp://', 'http://', 'https://'))
        # webcam = source.isnumeric() or source.endswith('.txt') or (is_url and not is_file)
        # if is_url and is_file:
        #     source = check_file(source)  # download

        # Directories
        # if not isinstance(self.yolo_weights, list):  # single yolo model
        #     exp_name = self.yolo_weights.stem
        # elif type(self.yolo_weights) is list and len(self.yolo_weights) == 1:  # single models after --yolo_weights
        #     exp_name = Path(self.yolo_weights[0]).stem
        # else:  # multiple models after --yolo_weights
        #     exp_name = 'ensemble'
        # exp_name = self.name if self.name else exp_name + "_" + self.reid_weights.stem
        # save_dir = increment_path(Path(self.project) / exp_name, exist_ok=self.exist_ok)  # increment run
        # (save_dir / 'tracks' if self.save_txt else save_dir).mkdir(parents=True, exist_ok=True)  # make dir

        # Load model
        self.device = select_device(self.device)
        self.is_seg = '-seg' in str(self.yolo_weights)
        self.model = DetectMultiBackend(self.yolo_weights, device=self.device, dnn=self.dnn, data=None, fp16=self.half)
        self.stride, self.names, self.pt = self.model.stride, self.model.names, self.model.pt
        self.model.warmup()
        # imgsz = check_img_size(self.imgsz, s=self.stride)  # check image size
        # Dataloader

        # if webcam:
        #     show_vid = check_imshow()
        #     dataset = LoadStreams(source, img_size=imgsz, stride=stride, auto=pt, vid_stride=vid_stride)
        #     nr_sources = len(dataset)
        # else:

        # dataset = LoadImages(self.source, img_size=imgsz, stride=self.stride, auto=self.pt)
        nr_sources = 1
        # vid_path, vid_writer, txt_path = [None] * nr_sources, [None] * nr_sources, [None] * nr_sources

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
            # visualize = increment_path()
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
            # if webcam:  # nr_sources >= 1
            #     p, im0, _ = path[i], im0s[i].copy(), dataset.count
            #     p = Path(p)  # to Path
            #     s += f'{i}: '
            #     txt_file_name = p.name
            #     save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            # else:
            #     p, im0, _ = path, im0s.copy(), getattr(dataset, 'frame', 0)
            #     p = Path(p)  # to Path
            #     # video file
            #     if source.endswith(VID_FORMATS):
            #         txt_file_name = p.stem
            #         save_path = str(save_dir / p.name)  # im.jpg, vid.mp4, ...
            #     # folder with imgs
            #     else:
            #         txt_file_name = p.parent.name  # get folder name containing current img
            #         save_path = str(save_dir / p.parent.name)  # im.jpg, vid.mp4, ...
            # curr_frames[i] = im0

            # 웹캠 사용 여부, 파일 경로, txt 파일 저장 부분 삭제
            im0 = img_to_infer.copy()

            # txt_path = str(save_dir / 'tracks' / txt_file_name)  # im.txt
            # s += '%gx%g ' % im.shape[2:]  # print string
            # imc = im0.copy() if save_crop else im0  # for save_crop

            # Tracking 하기 위해 프레임 저장
            self.curr_frames[i] = im0

            # annotator
            self.annotator = Annotator(im0, line_width=self.line_thickness, example=str(self.names))

            # 라벨 카운트 딕셔너리
            # self.label_count = defaultdict(int)

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
                
                # Pring results, txt 파일에 작성 하는 부분
                # for c in det[:, 5].unique():
                #     n = (det[:, 5] == c).sum()  # detections per class
                #     s += f"{n} {self.names[int(c)]}{'s' * (n > 1)}, "  # add to string

                # pass detections to strongsort
                with self.dt[3]:
                    self.outputs[i] = self.tracker_list[i].update(det.cpu(), im0)

                if len(self.outputs[i]) > 0:
                    for j, (output) in enumerate(self.outputs[i]):

                        bbox = output[0:4]
                        id = output[4]
                        cls = output[5]
                        conf = output[6]
                        
                        # txt 파일 필요 없음
                        # if self.save_txt:
                        #     # to MOT format
                        #     bbox_left = output[0]
                        #     bbox_top = output[1]
                        #     bbox_w = output[2] - output[0]
                        #     bbox_h = output[3] - output[1]
                        #     # Write MOT compliant results to file
                        #     with open(txt_path + '.txt', 'a') as f:
                        #         f.write(('%g ' * 10 + '\n') % (frame_idx + 1, id, bbox_left,  # MOT format
                        #                                        bbox_top, bbox_w, bbox_h, -1, -1, -1, i))
                        
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
                                # if save_crop:
                                #     txt_file_name = txt_file_name if (isinstance(path, list) and len(path) > 1) else ''
                                #     save_one_box(bbox.astype(np.int16), imc,
                                #                  file=save_dir / 'crops' / txt_file_name / names[

                                #                      c] / f'{id}' / f'{p.stem}.jpg', BGR=True)

                                # self.label_count[f'{self.names[c]}'] += 1

                                self.id_label_count_list.append([id, self.names[c]])
                            else:
                                continue
            else:
                pass
                # tracker_list[i].tracker.pred_n_update_all_tracks()

            im0 = self.annotator.result()
            # self.queue.put((im0, self.label_count, self.id_label_count_list))
            self.queue.put((im0, self.id_label_count_list))

            # print("show_vid : ", show_vid)
            # if show_vid:
            #     if platform.system() == 'Linux' and p not in windows:
            #         windows.append(p)
            #         cv2.namedWindow(str(p), cv2.WINDOW_NORMAL | cv2.WINDOW_KEEPRATIO)  # allow window resize (Linux)
            #         cv2.resizeWindow(str(p), im0.shape[1], im0.shape[0])
            #     cv2.imshow(str(p), im0)
            #     if cv2.waitKey(1) == ord('q'):  # 1 millisecond
            #         exit()

            # Save results (image with detections)
            # if save_vid:
            #     if vid_path[i] != save_path:  # new video
            #         vid_path[i] = save_path
            #         if isinstance(vid_writer[i], cv2.VideoWriter):
            #             vid_writer[i].release()  # release previous video writer
            #         if vid_cap:  # video
            #             fps = vid_cap.get(cv2.CAP_PROP_FPS)
            #             w = int(vid_cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            #             h = int(vid_cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            #         else:  # stream
            #             fps, w, h = 30, im0.shape[1], im0.shape[0]
            #         save_path = str(Path(save_path).with_suffix('.mp4'))  # force *.mp4 suffix on results videos
            #         vid_writer[i] = cv2.VideoWriter(save_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, (w, h))
            #     vid_writer[i].write(im0)

            self.prev_frames[i] = self.curr_frames[i]

            # Print total time (preprocessing + inference + NMS + tracking)
        LOGGER.info(
            f"{'' if len(det) else '(no detections), '}{sum([dt.dt for dt in self.dt if hasattr(dt, 'dt')]) * 1E3:.1f}ms")

        # Print results
        t = tuple(x.t / self.seen * 1E3 for x in self.dt)  # speeds per image
        LOGGER.info(
            f'Speed: %.1fms pre-process, %.1fms inference, %.1fms NMS, %.1fms {self.tracking_method} update per image at shape {(1, 3, *self.imgsz)}' % t)
        # if save_txt or save_vid:
        #     s = f"\n{len(list(save_dir.glob('tracks/*.txt')))} tracks saved to {save_dir / 'tracks'}" if save_txt else ''
        #     LOGGER.info(f"Results saved to {colorstr('bold', save_dir)}{s}")
        if self.update:
            strip_optimizer(self.yolo_weights)  # update model (to fix SourceChangeWarning)


    def get_data(self):
        return self.queue.get()
