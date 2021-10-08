from typing import List, Union
from dataclasses import dataclass
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from pathlib import Path
from collections import defaultdict

from dna.det import Detection
from dna.types import BBox
from . import Track, TrackState, ObjectTracker, utils


class TrackerCallback(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self, tracker: ObjectTracker) -> None: pass

    @abstractmethod
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    @abstractmethod
    def tracked(self, tracker :ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None: pass


class DemuxTrackerCallback(TrackerCallback):
    def __init__(self, callbacks: List[TrackerCallback]) -> None:
        super().__init__()
        self.callbacks = callbacks

    def track_started(self, tracker) -> None:
        for cb in self.callbacks:
            with suppress(Exception): cb.track_started(tracker)

    def track_stopped(self, tracker) -> None:
        for cb in self.callbacks:
            cb.track_stopped(tracker)
            # with suppress(Exception): cb.track_stopped(tracker)

    def tracked(self, tracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for cb in self.callbacks:
            cb.tracked(tracker, frame, frame_idx, tracks)
            # with suppress(Exception): cb.tracked(tracker, frame, frame_idx, tracks)


class TrailCollector(TrackerCallback):
    @dataclass(frozen=True, unsafe_hash=True)
    class Mark:
        state: TrackState
        location: BBox

        def __repr__(self) -> str:
            return f"[{self.location}]-"

    def __init__(self) -> None:
        super().__init__()
        self.tracks = defaultdict(list)

    def get_trail(self, track_id: str, def_value=None) -> Union[List[Mark], None]:
        return self.tracks.get(track_id, def_value)

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                mark = TrailCollector.Mark(state=track.state, location=track.location)
                self.tracks[track.id].append(mark)
            elif track.state == TrackState.Deleted:
                self.tracks.pop(track.id, None)


class TrackWriter(TrackerCallback):
    def __init__(self, track_file: Path) -> None:
        super().__init__()
        self.track_file = track_file
        self.out_handle = None

    def track_started(self) -> None:
        super().track_started()
        self.out_handle = open(self.track_file, 'w')

    def track_stopped(self) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None
        super().track_stopped()

    def tracked(self, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            self.out_handle.write(track.to_string() + '\n')