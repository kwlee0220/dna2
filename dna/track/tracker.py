from typing import List
from abc import ABCMeta, abstractmethod
import logging

from dna.det import Detection
from .types import Track


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