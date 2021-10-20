from typing import List, Union
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from pathlib import Path

from .types import Track, TrackState
from .tracker import ObjectTracker


class TrackerCallback(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self, tracker: ObjectTracker) -> None: pass

    @abstractmethod
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    @abstractmethod
    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None: pass


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


class TrackWriter(TrackerCallback):
    def __init__(self, track_file: Path) -> None:
        super().__init__()
        self.track_file = track_file
        self.out_handle = None

    def track_started(self, tracker: ObjectTracker) -> None:
        super().track_started(tracker)
        self.out_handle = open(self.track_file, 'w')
    
    def track_stopped(self, tracker: ObjectTracker) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None
        super().track_stopped(tracker)

    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            self.out_handle.write(track.to_string() + '\n')