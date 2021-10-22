from typing import List
from pathlib import Path
from urllib.parse import parse_qs
import logging

import yaml
import numpy as np
import cv2
import torch
import torch.backends.cudnn as cudnn

import sys
FILE = Path(__file__).absolute()
_YOLOV5_DIR = str(Path(FILE.parents[3], 'dna-plugins/dna-yolov5-torch'))
if Path(_YOLOV5_DIR).exists() and not _YOLOV5_DIR in sys.path:
    sys.path.append(_YOLOV5_DIR)

from dna import get_logger
from dna.utils import parse_query
from dna.det import ObjectDetector, Box, Detection
from dna.utils import get_first_param

from models.experimental import attempt_load
from utils.general import check_img_size, non_max_suppression, scale_coords
from utils.torch_utils import select_device
from utils.augmentations import letterbox


_LOGGER = get_logger("dna.det")

def load(query: str):
    args = parse_query(query)

    with open('./configurations.yaml') as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    model_id = args.get('model', 's')
    model = models['detector']['yolov5'][model_id]
    weights_file = Path(model['weights'])

    score = args.get('score')
    score = score if score else model.get('score')
    score = float(score) if score else None

    _LOGGER.info(f'Loading Yolov5Detector: model={model_id}, weights={weights_file.absolute()}')
    
    return Yolov5Detector(weights_file, conf_thres=score)


class Yolov5Detector(ObjectDetector):
    def __init__(self,
                weights_file: Path = None,
                imgsz: int=640, 
                conf_thres: float=0.25, # confidence threshold
                iou_thres: float=0.45,  # NMS IOU threshold
                max_det: int=1000,      # maximum detections per image
                device: str='',         # cuda device, i.e. 0 or 0,1,2,3 or cpu
                classes=None,           # filter by class: --class 0, or --class 0 2 3
                agnostic_nms: bool=False  # class-agnostic NMS
                ) -> None:
        self.weights = weights_file
        self.device = select_device(device)
        self.img_sz = imgsz
        self.conf_thres = conf_thres if conf_thres else 0.25
        self.iou_thres = iou_thres
        self.max_det = max_det
        self.classes = classes
        self.agnostic_nms = agnostic_nms

        self.model = attempt_load(self.weights, map_location=self.device)  # load FP32 model
        self.stride = int(self.model.stride.max())  # model stride
        self.names = [f'class{i}' for i in range(1000)]  # assign defaults
        self.names = self.model.module.names if hasattr(self.model, 'module') else self.model.names  # get class names
        imgsz = check_img_size(imgsz, s=self.stride)  # check image size

        if self.device.type != 'cpu':
            self.model(torch.zeros(1, 3, imgsz, imgsz).to(self.device).type_as(next(self.model.parameters())))  # run once

    @torch.no_grad()
    def detect(self, frame, frame_idx=-1) -> List[Detection]:
        # Padded resize
        img = letterbox(frame, self.img_sz, stride=self.stride, auto=True)[0]

        # Convert
        img = img.transpose((2, 0, 1))[::-1]  # HWC to CHW, BGR to RGB
        img = np.ascontiguousarray(img)

        img = torch.from_numpy(img).to(self.device)
        img = img.float()  # uint8 to fp16/32
        img /= 255.0  # 0 - 255 to 0.0 - 1.0
        if len(img.shape) == 3:
            img = img[None]  # expand for batch dim

        # inference
        preds_batch = self.model(img)[0]

        # NMS
        preds_batch = non_max_suppression(preds_batch, self.conf_thres, self.iou_thres, self.classes,
                                        self.agnostic_nms, max_det=self.max_det)

        det_list = []
        for i, preds in enumerate(preds_batch):  # detections per image
            # Rescale boxes from img_size to im0 size
            preds[:, :4] = scale_coords(img.shape[2:], preds[:, :4], frame.shape)
            for pred in preds:
                det = pred.tolist()

                bbox = Box.from_tlbr(det[:4])
                det_list.append(Detection(bbox, self.names[int(det[5])], det[4]))

        return det_list