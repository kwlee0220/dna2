from datetime import datetime
from typing import List, Union
from pathlib import Path
import sys

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
from . import Track, TrackState, DetectionBasedObjectTracker
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState


class DeepSORTTracker(DetectionBasedObjectTracker):
    # def __init__(self, detector: ObjectDetector, domain: BBox, weights_file, det_dict = None,
    #                 min_detection_score=0, matching_threshold=0.5, max_iou_distance=0.7, max_age=30) -> None:
    def __init__(self, detector: ObjectDetector, domain: Box, tracker_conf: OmegaConf,
                    blind_regions=None) -> None:
        super().__init__()

        self.__detector = detector
        self.det_dict = tracker_conf.det_mapping
        self.domain = domain
        self.min_size = Size2d.from_np(tracker_conf.min_size)
        self.min_detection_score = tracker_conf.min_detection_score

        wt_path = Path(tracker_conf.model_file)
        if not wt_path.is_absolute():
            wt_path = utils.get_dna_home_dir() / wt_path

        self.deepsort = deepsort_rbc(domain = domain,
                                    wt_path=wt_path,
                                    matching_threshold=tracker_conf.matching_threshold,
                                    max_iou_distance=tracker_conf.max_iou_distance,
                                    max_age=int(tracker_conf.max_age),
                                    n_init=int(tracker_conf.n_init),
                                    min_size=self.min_size,
                                    blind_regions=blind_regions)
        self.blind_regions = blind_regions
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

    def track(self, frame, frame_idx:int, ts:datetime) -> List[Track]:
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

        # 일정 점수 이하의 detection들은 무시한다.
        detections = [det for det in dets if det.score >= self.min_detection_score]

        self.__last_frame_detections = detections
        boxes, scores = self.split_boxes_scores(self.__last_frame_detections)

        tracker, deleted_tracks = self.deepsort.run_deep_sort(frame.astype(np.uint8), boxes, scores)

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

    def split_boxes_scores(self, det_list):
        boxes = []
        scores = []
        for det in det_list:
            boxes.append(det.bbox.tlwh)
            scores.append(det.score)
        
        return np.array(boxes), scores