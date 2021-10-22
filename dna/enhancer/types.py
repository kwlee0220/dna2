from __future__ import annotations
from typing import List
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dna import Point, Box, utils
from dna.track.types import Track


@dataclass(frozen=True, unsafe_hash=True)
class TrackEvent:
    camera_id: str
    luid: int
    location: Box
    frame_index: int
    ts: float
    
    def __repr__(self) -> str:
        ts = int(round(self.ts * 1000))
        return (f"TrackEvent[cam={self.camera_id}, id={self.luid}, loc=={self.location}, "
                f"frame={self.frame_index}, ts={ts}]")