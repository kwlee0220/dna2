from __future__ import annotations
from typing import List
from dataclasses import dataclass
from datetime import datetime

import numpy as np
import cv2

from dna import Box, Size2d, plot_utils

@dataclass(frozen=True, unsafe_hash=True)
class DeepSORTParams:
    metric_threshold: float
    max_iou_distance: float
    n_init: int
    max_age: int
    min_size: Size2d
    max_overlap_ratio: float
    blind_zones: List[Box]
    dim_zones: List[Box]

from enum import Enum
class TrackState(Enum):
    Null = 0
    Tentative = 1
    Confirmed = 2
    TemporarilyLost = 3
    Deleted = 4


@dataclass(frozen=True, unsafe_hash=True)
class Track:
    id: int
    state: TrackState
    location: Box
    frame_index: int
    ts: float

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
    
    def __repr__(self) -> str:
        epoch = int(round(self.ts * 1000))
        return f"{self.state.name}[{self.id}]={self.location}, frame={self.frame_index}, ts={epoch}"

    def draw(self, mat, color, label_color=None, line_thickness=2) -> np.ndarray:
        loc = self.location

        mat = loc.draw(mat, color, line_thickness=line_thickness)
        mat = cv2.circle(mat, loc.center().xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            mat = plot_utils.draw_label(mat, str(self.id), loc.tl.astype(int), label_color, color, 2)

        return mat

    def to_string(self) -> str:
        tlbr = self.location.to_tlbr()
        epoch = int(round(self.ts * 1000))
        return (f"{self.frame_index},{self.id},{tlbr[0]:.0f},{tlbr[1]:.0f},{tlbr[2]:.0f},{tlbr[3]:.0f},"
                f"{self.state.value},{epoch}")
    
    @staticmethod
    def from_string(csv) -> Track:
        parts = csv.split(',')

        frame_idx = int(parts[0])
        track_id = int(parts[1])
        tlbr = np.array(parts[2:6]).astype(int)
        bbox = Box.from_tlbr(tlbr)
        state = TrackState(int(parts[6]))
        ts = int(parts[7]) / 1000
        
        return Track(id=track_id, state=state, location=bbox, frame_index=frame_idx, ts=ts)