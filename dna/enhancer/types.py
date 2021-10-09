from __future__ import annotations
from typing import List
from dataclasses import dataclass
from datetime import datetime

import numpy as np

from dna import Point, BBox, utils

@dataclass(frozen=True, unsafe_hash=True)
class TrackEvent:
    camera_id: str
    luid: str
    location: BBox
    frame_index: int
    ts: datetime
    
    def __repr__(self) -> str:
        ts_str = utils.datetime2str(self.ts)
        return (f"TrackEvent[cam={self.camera_id}, id={self.luid}, loc=={self.location}, "
                f"frame={self.frame_index}, ts={ts_str}]")

@dataclass(frozen=True, unsafe_hash=True)
class TimedPoint:
    location: BBox
    ts: int

@dataclass(frozen=True, unsafe_hash=True)
class Trajectory:
    camera_id: str
    luid: str
    path: List[TimedPoint]
    length: float
    begin_ts: int
    end_ts: int

    @staticmethod
    def from_track_events(events: List[TrackEvent]) -> Trajectory:
        first = events[0]
        last = events[-1]
        path = [TimedPoint(location=ev.location, ts=ev.ts) for ev in events]
        length = sum([Point.distance(tpt1.location.center, tpt2.location.center) \
                        for tpt1, tpt2 in zip(path, path[1:])])

        return Trajectory(camera_id=first.camera_id, luid=first.luid,
                            path=path, length=length,
                            begin_ts=first.ts, end_ts=last.ts)