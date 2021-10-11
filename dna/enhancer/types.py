from __future__ import annotations
from typing import List
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dna import Point, BBox, utils
from dna.track.types import Track


@dataclass(frozen=True, unsafe_hash=True)
class TrackEvent:
    camera_id: str
    luid: int
    location: BBox
    frame_index: int
    ts: datetime
    
    def __repr__(self) -> str:
        ts_str = utils.datetime2str(self.ts)
        return (f"TrackEvent[cam={self.camera_id}, id={self.luid}, loc=={self.location}, "
                f"frame={self.frame_index}, ts={ts_str}]")