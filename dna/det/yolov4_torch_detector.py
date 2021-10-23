from re import L
from typing import List
from pathlib import Path
from urllib.parse import parse_qs
import logging

import yaml
import numpy as np
import cv2

import sys
FILE = Path(__file__).absolute()
_YOLOV4_DIR = str(Path(FILE.parents[3], 'dna-plugins/dna-yolov4-torch'))
if not _YOLOV4_DIR in sys.path:
    sys.path.append(_YOLOV4_DIR)

from dna import Box, get_logger
from dna.utils import parse_query, get_dna_home_dir
from dna.det import Detection, ObjectDetector
from dna.utils import get_first_param

from dna_yolov4_torch.tool.utils import load_class_names
from dna_yolov4_torch.tool.torch_utils import do_detect
from dna_yolov4_torch.tool.darknet2pytorch import Darknet

_LOGGER = get_logger("dna.det")
def load(query: str, conf_home:Path =None):
    args = parse_query(query)
    model_id = args.get('model', 'normal')

    conf_path = get_dna_home_dir(conf_home) / "configurations.yaml"
    with open(conf_path) as f:
        models = yaml.load(f, Loader=yaml.FullLoader)

    model = models['detector']['yolov4'][model_id]
    cfg_file = Path(model['cfg'])
    weights_file = Path(model['weights'])
    class_names_file = Path(model['class_names'])
    class_names = load_class_names(class_names_file)

    score = args.get('score')
    score = score if score else model.get('score')
    score = float(score) if score else None

    _LOGGER.info((f'Loading Yolov4TorchDetector: cfg={cfg_file.absolute()}, '
                    f'weights={weights_file.absolute()}, class_file={class_names_file.absolute()}'))
    return Yolov4TorchDetector(cfg_file, weights_file, class_names)

def _load_weights(model, file: Path):
    if not file.exists():
        import gdown
        if file.name == 'yolov4.weights':
            url = 'https://drive.google.com/uc?id=17SKxQtvhpVQbmlUP4n1yBP-0lyYVkgty'
            gdown.download(url, file.as_posix(), quiet=False)
        if file.name == 'yolov4-tiny.weights':
            url = 'https://drive.google.com/uc?id=12AdGKIZqAUIZtTLxOvFMaS-wMkEp_THx'
            gdown.download(url, file.as_posix(), quiet=False)
    
    model.load_weights(file)

class Yolov4TorchDetector(ObjectDetector):
    def __init__(self,
                cfg_file,
                weights_file,
                class_names,
                conf_thres=0.4,     # confidence threshold
                nms_thres=0.6,      # NMS IOU threshold
                use_cuda = True
                ) -> None:
        self.model = Darknet(cfg_file)
        _load_weights(self.model, weights_file)
        # self.model.load_weights(weights_file)

        self.num_classes = self.model.num_classes
        self.class_names = class_names
        self.conf_thres = conf_thres
        self.nms_thres = nms_thres

        self.use_cuda = use_cuda
        if self.use_cuda:
            self.model.cuda()

    def detect(self, frame, frame_index=-1) -> List[Detection]:
        sized = cv2.resize(frame, (self.model.width, self.model.height))
        sized = cv2.cvtColor(sized, cv2.COLOR_BGR2RGB)

        h, w, _ = frame.shape
        batched_boxes = do_detect(self.model, sized, self.conf_thres, self.nms_thres, self.use_cuda)

        return [self.box_to_detection(box, w, h) for box in batched_boxes[0]]

    def box_to_detection(self, box, w, h):
        coords = np.array([box[0] * w, box[1] * h, box[2] * w, box[3] * h])
        bbox = Box.from_tlbr(coords)
        conf = box[5]
        label = self.class_names[box[6]]
        return Detection(bbox, label=label, score=conf)