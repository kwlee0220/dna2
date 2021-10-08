from __future__ import annotations
from typing import List
from dataclasses import dataclass

import numpy as np
import cv2

from dna import BBox, plot_utils

from enum import Enum
class TrackState(Enum):
    Null = 0
    Tentative = 1
    Confirmed = 2
    TemporarilyLost = 3
    Deleted = 4


@dataclass(frozen=True, unsafe_hash=True)
class Track:
    id: str
    state: TrackState
    location: BBox
    frame_index: int
    utc_epoch: int

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
    
    def __repr__(self) -> str:
        return f"{self.state.name}[{self.id}]={self.location}, frame={self.frame_index}, ts={self.utc_epoch}"

    def draw(self, mat, color, label_color=None, line_thickness=2) -> np.ndarray:
        loc = self.location

        mat = loc.draw(mat, color, line_thickness=line_thickness)
        mat = cv2.circle(mat, loc.center.xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            mat = plot_utils.draw_label(mat, self.id, loc.tl.astype(int), label_color, color, 2)

        return mat

    def to_string(self) -> str:
        tlbr = self.location.tlbr
        return (f"{self.frame_index},{self.id},{tlbr[0]:.3f},{tlbr[1]:.3f},{tlbr[2]:.3f},{tlbr[3]:.3f},"
                f"{self.state.value},{self.utc_epoch}")
    
    @staticmethod
    def from_string(csv) -> Track:
        parts = csv.split(',')

        frame_idx = int(parts[0])
        track_id = parts[1]
        tlbr = np.array(parts[2:6]).astype(float)
        bbox = BBox.from_tlbr(tlbr)
        state = TrackState(int(parts[6]))
        utc_epoch = int(parts[7])
        
        return Track(id=track_id, state=state, location=bbox, frame_index=frame_idx, utc_epoch=utc_epoch)