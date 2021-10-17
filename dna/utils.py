from datetime import datetime, timezone
from time import time
from typing import Tuple
from pathlib import Path

from dna.types import BBox

def datetime2utc(dt: datetime) -> int:
    return int(dt.replace(tzinfo=timezone.utc).timestamp())

def utc2datetime(ts: int) -> datetime:
    return datetime.fromtimestamp(ts / 1000)

def datetime2str(dt: datetime) -> str:
    return dt.strftime("%Y-%m-%d %H:%M:%S.%f")

def utc_now() -> int:
    return int(time() * 1000)

def _parse_keyvalue(kv) -> Tuple[str,str]:
    pair = kv.split('=')
    if len(pair) == 2:
        return tuple(pair)
    else:
        return pair, None

def parse_query(query):
    if not query or len(query) == 0:
        return dict()
    return dict([_parse_keyvalue(kv) for kv in query.split(':')])

def get_first_param(args, key, def_value=None):
    value = args.get(key)
    return value[0] if value else def_value

def get_first_param_path(args, key, def_value=None):
    value = args.get(key)
    if value:
        return Path(value[0])
    elif def_value:
        return Path(def_value)
    else:
        return None


import logging
_LOGGERS = dict()
_LOG_FORMATTER = logging.Formatter("%(levelname)s: %(message)s (%(filename)s)")

def get_logger(name=None):
    logger = _LOGGERS.get(name)
    if not logger:
        logger = logging.getLogger(name)
        _LOGGERS[name] = logger
        
        logger.setLevel(logging.DEBUG)

        console = logging.StreamHandler()
        # console.setLevel(logging.INFO)
        console.setFormatter(_LOG_FORMATTER)
        logger.addHandler(console)
        
    return logger


from dna import color, plot_utils
import cv2
import numpy as np

def draw_boxes(convas, boxes, box_color, label_color=None, line_thickness=2):
    for idx, box in enumerate(boxes):
        box.draw(convas, box_color)
        if label_color:
            msg = f"{idx:02d}"
            mat = plot_utils.draw_label(convas, msg, box.tl.astype(int), label_color, box_color, 2)
    return convas

def draw_ds_tracks(convas, tracks, box_color, label_color=None, line_thickness=2):
    for idx, track in enumerate(tracks):
        box = BBox.from_tlbr(track.to_tlbr())
        box.draw(convas, box_color)
        if label_color:
            msg = f"{track.track_id}[{track.state}]"
            mat = plot_utils.draw_label(convas, msg, box.br.astype(int), label_color, box_color, 2)
    return convas

def draw_ds_detections(convas, dets, box_color, label_color=None, line_thickness=2):
    for idx, det in enumerate(dets):
        box = BBox.from_tlbr(det.to_tlbr())
        box.draw(convas, box_color)
        if label_color:
            msg = f"{idx:02d}"
            mat = plot_utils.draw_label(convas, msg, box.tl.astype(int), label_color, box_color, 2)
    return convas

def find_track_index(track_id, tracks, track_indices):
    for idx, tidx in enumerate(track_indices):
        if tracks[tidx].track_id == track_id:
            return idx
    return -1