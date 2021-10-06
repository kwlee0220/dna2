from typing import List
from pathlib import Path

import numpy as np
import cv2

import sys
FILE = Path(__file__).absolute()
DEEPSORT_DIR = str(FILE.parents[0] / 'deepsort')
if not DEEPSORT_DIR in sys.path:
    sys.path.append(DEEPSORT_DIR)

from dna import BBox
from dna.det import Detection
from . import Track, TrackState, ObjectTracker
from .deepsort.deepsort import deepsort_rbc
from .deepsort.track import Track as DSTrack
from .deepsort.track import TrackState as DSTrackState


class DeepSORTTracker(ObjectTracker):
    def __init__(self, weights_file, matching_threshold=0.5, max_iou_distance=0.7, max_age=30) -> None:
        self.deepsort = deepsort_rbc(wt_path=weights_file.absolute(),
                                    matching_threshold=matching_threshold,
                                    max_iou_distance=max_iou_distance,
                                    max_age=max_age)
        self.tracks = {}

    def track(self, mat, frame_idx:int, det_list: List[Detection]) -> List[Track]:
        boxes, scores = self.split_boxes_scores(det_list)
        tracker, deleted_tracks = self.deepsort.run_deep_sort(mat.astype(np.uint8), boxes, scores)

        tracks = []
        for ds_track in deleted_tracks:
            track = self.tracks.pop(ds_track.track_id, None)
            if track:
                tracks.append(track)

        for ds_track in tracker.tracks:
            location = BBox(ds_track.to_tlwh())
            track = self.tracks.get(ds_track.track_id, None)
            trail = track.location_trail + [location] if track else [location]
            track = self.to_dna_track(ds_track, trail, frame_idx)
            self.tracks[ds_track.track_id] = track
            tracks.append(track)
        
        return tracks

    # @property
    # def tracks(self) -> List[Track]:
    #     return self.__tracks

    def to_dna_track(self, ds_track: DSTrack, trail: List[BBox], frame_idx: int) -> Track:
        if ds_track.state == DSTrackState.Confirmed:
            state = TrackState.Confirmed if ds_track.time_since_update <= 0 else TrackState.TemporarilyLost
        elif ds_track.state == DSTrackState.Tentative:
            state = TrackState.Tentative
        elif ds_track.state == DSTrackState.Deleted:
            state = TrackState.Deleted
        return Track(id=str(ds_track.track_id), state=state, location=BBox(ds_track.to_tlwh()),
                    location_trail=trail, frame_idx=frame_idx)

    def split_boxes_scores(self, det_list):
        boxes = []
        scores = []
        for det in det_list:
            boxes.append(det.bbox.tlwh)
            scores.append(det.score)
        
        return np.array(boxes), scores