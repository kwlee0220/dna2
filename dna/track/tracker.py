from datetime import datetime
from typing import List
from abc import ABCMeta, abstractmethod
from contextlib import suppress
from pathlib import Path
import logging

import numpy as np

from dna import Box
from dna.det import Detection
from dna.det.detector import ObjectDetector
from . import Track, TrackState


class ObjectTracker(metaclass=ABCMeta):
    logger = logging.getLogger("dna.track.tracker")
    logger.setLevel(logging.INFO)

    @abstractmethod
    def track(self, frame, frame_idx:int, ts) -> List[Track]: pass


class DetectionBasedObjectTracker(ObjectTracker):
    @property
    @abstractmethod
    def detector(self) -> ObjectDetector: pass

    @abstractmethod
    def last_frame_detections(self) -> List[Detection]: pass


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

    def track(self, frame, frame_idx:int, ts:datetime) -> List[Track]:
        if not self.look_ahead:
            return []

        if self.look_ahead.frame_index > frame_idx:
            return []

        # throw track event lines upto target_idx -
        while self.look_ahead.frame_index < frame_idx:
            self.look_ahead = self._look_ahead()

        tracks = []
        while self.look_ahead.frame_index == frame_idx:
            tracks.append(self.look_ahead)

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead is None:
                break

        return tracks
        
    def _look_ahead(self) -> Track:
        line = self.__file.readline().rstrip()
        if line:
            return Track.from_string(line)
        else:
            self.__file.close()
            return None

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"