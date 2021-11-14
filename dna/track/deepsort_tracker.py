from datetime import datetime
from typing import List, Union, Tuple
from pathlib import Path
from dataclasses import dataclass
import sys
from enum import Enum

import numpy as np
import cv2
from omegaconf.omegaconf import OmegaConf
import logging

FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

from dna import Box, Size2d, utils, get_logger
from dna.det import ObjectDetector, Detection
from . import Track, TrackState, DetectionBasedObjectTracker, DeepSORTParams
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState

class DeepSORTTracker(DetectionBasedObjectTracker):
    def __init__(self, detector: ObjectDetector, domain: Box, tracker_conf: OmegaConf) -> None:
        super().__init__()

        self.__detector = detector
        self.det_dict = tracker_conf.det_mapping
        self.domain = domain
        self.min_detection_score = tracker_conf.min_detection_score

        wt_path = Path(tracker_conf.model_file)
        if not wt_path.is_absolute():
            wt_path = utils.get_dna_home_dir() / wt_path

        if tracker_conf.get("blind_zones", None):
            blind_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in tracker_conf.blind_zones]
        else:
            blind_zones = []
        if tracker_conf.get("dim_zones", None):
            dim_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in tracker_conf.dim_zones]
        else:
            dim_zones = []

        self.params = DeepSORTParams(metric_threshold=tracker_conf.metric_threshold,
                                max_iou_distance=tracker_conf.max_iou_distance,
                                max_age=int(tracker_conf.max_age),
                                n_init=int(tracker_conf.n_init),
                                max_overlap_ratio = tracker_conf.max_overlap_ratio,
                                min_size=Size2d.from_np(tracker_conf.min_size),
                                blind_zones=blind_zones,
                                dim_zones=dim_zones)
        self.deepsort = deepsort_rbc(domain = domain,
                                    wt_path=wt_path,
                                    params=self.params)
        self.__last_frame_detections = []

        level_name = tracker_conf.get("log_level", "info").upper()
        level = logging.getLevelName(level_name)
        logger = get_logger("dna.track.deep_sort")
        logger.setLevel(level)
        
    @property
    def detector(self) -> ObjectDetector:
        return self.__detector

    def last_frame_detections(self) -> List[Detection]:
        return self.__last_frame_detections

    def __replace_detection_label(self, det) -> Union[Detection,None]:
        label = self.det_dict.get(det.label, None)
        if label:
            return Detection(det.bbox, label, det.score)
        else:
            return None

    def track(self, frame, frame_idx:int, ts) -> List[Track]:
        # detector를 통해 match 대상 detection들을 얻는다.
        dets = self.detector.detect(frame, frame_index=frame_idx)

        # 검출 물체 중 관련있는 label의 detection만 사용한다.
        if self.det_dict:
            new_dets = []
            for det in dets:
                label = self.det_dict.get(det.label, None)
                if label:
                    new_dets.append(Detection(det.bbox, label, det.score))
            dets = new_dets

        # 일정 점수 이하의 detection들과 blind zone에 포함된 detection들은 무시한다.
        def is_valid_detection(det):
            return det.score >= self.min_detection_score and \
                    not any(zone.contains(det.bbox) for zone in self.params.blind_zones)
        detections = [det for det in dets if is_valid_detection(det)]

        self.__last_frame_detections = detections
        bboxes, scores = self.split_boxes_scores(self.__last_frame_detections)
        tracker, deleted_tracks = self.deepsort.run_deep_sort(frame.astype(np.uint8), bboxes, scores)

        active_tracks = [self.to_dna_track(ds_track, frame_idx, ts) for ds_track in tracker.tracks]
        deleted_tracks = [self.to_dna_track(ds_track, frame_idx, ts) for ds_track in deleted_tracks]
        return active_tracks + deleted_tracks

    def to_dna_track(self, ds_track: DSTrack, frame_idx: int, ts:datetime) -> Track:
        if ds_track.state == DSTrackState.Confirmed:
            state = TrackState.Confirmed if ds_track.time_since_update <= 0 else TrackState.TemporarilyLost
        elif ds_track.state == DSTrackState.Tentative:
            state = TrackState.Tentative
        elif ds_track.state == DSTrackState.Deleted:
            state = TrackState.Deleted

        return Track(id=ds_track.track_id, state=state,
                    location=Box.from_tlbr(np.rint(ds_track.to_tlbr())),
                    frame_index=frame_idx, ts=ts)

    def split_boxes_scores(self, det_list) -> Tuple[List[Box], List[float]]:
        boxes = []
        scores = []
        for det in det_list:
            boxes.append(det.bbox)
            scores.append(det.score)
        
        return np.array(boxes), scores