from datetime import datetime
from typing import List
from pathlib import Path

import numpy as np
import cv2

import sys

from dna.det.detector import ObjectDetector
FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

from dna import BBox
from dna.det import ObjectDetector, Detection
from . import Track, TrackState, DetectionBasedObjectTracker
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState


class DeepSORTTracker(DetectionBasedObjectTracker):
    def __init__(self, detector: ObjectDetector, weights_file,
                    matching_threshold=0.5, max_iou_distance=0.7, max_age=30) -> None:
        super().__init__()

        self.__detector = detector
        self.deepsort = deepsort_rbc(wt_path=weights_file.absolute(),
                                    matching_threshold=matching_threshold,
                                    max_iou_distance=max_iou_distance,
                                    max_age=max_age)
        self.__last_frame_detections = []
        
    @property
    def detector(self) -> ObjectDetector:
        return self.__detector

    def last_frame_detections(self) -> List[Detection]:
        return self.__last_frame_detections

    def track(self, frame, frame_idx:int, ts:datetime) -> List[Track]:
        self.__last_frame_detections = self.detector.detect(frame, frame_index=frame_idx)
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

        return Track(id=ds_track.track_id, state=state, location=BBox(ds_track.to_tlwh()),
                    frame_index=frame_idx, ts=ts)

    def split_boxes_scores(self, det_list):
        boxes = []
        scores = []
        for det in det_list:
            boxes.append(det.bbox.tlwh)
            scores.append(det.score)
        
        return np.array(boxes), scores