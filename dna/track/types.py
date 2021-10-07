from typing import List
from dataclasses import dataclass

import numpy as np
import cv2

from dna import BBox, plot_utils
from dna.det import Detection

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

    def is_tentative(self) -> bool:
        return self.state == TrackState.Tentative

    def is_confirmed(self) -> bool:
        return self.state == TrackState.Confirmed

    def is_temporarily_lost(self) -> bool:
        return self.state == TrackState.TemporarilyLost

    def is_deleted(self) -> bool:
        return self.state == TrackState.Deleted
    
    def __repr__(self) -> str:
        length = len(self.location_trail)
        return f"{self.state.name}[{self.id}:len={length}]={self.location}"

    def draw(self, mat, color, label_color=None, line_thickness=2) -> np.ndarray:
        loc = self.location

        mat = loc.draw(mat, color, line_thickness=line_thickness)
        mat = cv2.circle(mat, loc.center.xy.astype(int), 4, color, thickness=-1, lineType=cv2.LINE_AA)
        if label_color:
            mat = plot_utils.draw_label(mat, self.id, loc.tl.astype(int), label_color, color, 2)
        return mat