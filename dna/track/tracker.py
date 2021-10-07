from typing import List
from abc import ABCMeta, abstractmethod
import logging
from pathlib import Path

import numpy as np

from dna import BBox
from dna.det import Detection
from . import Track, TrackState


class TrackerCallback(metaclass=ABCMeta):
    @abstractmethod
    def track_started(self) -> None: pass

    @abstractmethod
    def track_stopped(self) -> None: pass

    @abstractmethod
    def tracked(self, frame, frame_idx: int, detections: List[Detection], tracks: List[Track]) -> None: pass


class ObjectTracker(metaclass=ABCMeta):
    logger = logging.getLogger("dna.track.tracker")
    logger.setLevel(logging.INFO)

    @abstractmethod
    def track(self, mat, frame_idx:int, det_list: List[Detection]) -> List[Track]: pass

    # @property
    # @abstractmethod
    # def tracks(self) -> List[Track] : pass


class LogFileBasedObjectTracker(ObjectTracker):
    def __init__(self, track_file: Path) -> None:
        """[Create an ObjectTracker object that issues tracking events from a tracking log file.]

        Args:
            det_file (Path): Path to the detection file.
        """
        self.__file = open(track_file, 'r')
        self.look_ahead = self._look_ahead()

    @property
    def file(self) -> Path:
        return self.__file

    def track(self, mat, frame_idx:int, det_list: List[Detection]) -> List[Track]:
        if not self.look_ahead:
            return []

        if self.look_ahead.frame_idx > frame_idx:
            return []

        # throw track event lines upto target_idx -
        while self.look_ahead.frame_idx < frame_idx:
            self.look_ahead = self._look_ahead()

        tracks = []
        while self.look_ahead.frame_idx == frame_idx:
            tracks.append(self.look_ahead)

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead is None:
                break

        return tracks
        
    def _look_ahead(self):
        line = self.__file.readline().rstrip()
        if line:
            return self._parse_line(line.split(','))
        else:
            self.__file.close()
            return None

    def _parse_line(self, parts: List[str]):
        frame_idx = int(parts[0])
        track_id = parts[1]
        tlbr = np.array(parts[2:6]).astype(float)
        bbox = BBox.from_tlbr(tlbr)
        state = TrackState(int(parts[6]))
        return Track(track_id, state, bbox, frame_idx, [])

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"