from __future__ import annotations
from typing import List
from dataclasses import dataclass
from pathlib import Path
from abc import ABCMeta, abstractmethod
import logging

import numpy as np

from dna import Box
from dna.det import Detection


class ObjectDetector(metaclass=ABCMeta):
    logger = logging.getLogger("dna.det.detector")
    logger.setLevel(logging.INFO)

    @abstractmethod
    def detect(self, frame: np.ndarray, frame_index: int=-1) -> List[Detection]:
        """Detect objects from the image and returns their locations

        Args:
            frame ([np.ndarray]): an image from OpenCV
            frame_index (int, optional): frame index. Defaults to -1.

        Returns:
            List[Detection]: a list of Detection objects
        """
        pass

    # def detect_images(self, mats):
    #     return [self.detect(mat) for mat in mats]


class LogReadingDetector(ObjectDetector):
    def __init__(self, det_file: Path) -> None:
        """[Create an ObjectDetector object that issues detections from a detection file.]

        Args:
            det_file (Path): Path to the detection file.
        """
        self.__file = open(det_file, 'r')
        self.look_ahead = self._look_ahead()

    @property
    def file(self) -> Path:
        return self.__file

    def detect(self, frame, frame_index: int) -> List[Detection]:
        if not self.look_ahead:
            return []

        idx = int(self.look_ahead[0])
        if idx > frame_index:
            return []

        # throw detection lines upto target_idx -
        while idx < frame_index:
            self.look_ahead = self._look_ahead()
            idx = int(self.look_ahead[0])

        detections = []
        while idx == frame_index and self.look_ahead:
            detections.append(self._parse_line(self.look_ahead))

            # read next line
            self.look_ahead = self._look_ahead()
            if self.look_ahead:
                idx = int(self.look_ahead[0])
            else:
                idx += 1

        return detections

    def _look_ahead(self):
        line = self.__file.readline().rstrip()
        if line:
            return line.split(',')
        else:
            self.__file.close()
            return None

    def _parse_line(self, parts):
        tlbr = np.array(parts[2:6]).astype(float)
        bbox = Box.from_tlbr(tlbr)
        label = parts[10] if len(parts) >= 11 else None
        return Detection(bbox=bbox, label=label, score=float(parts[6]))

    def __repr__(self) -> str:
        current_idx = int(self.look_ahead[0]) if self.look_ahead else -1
        return f"{self.__class__.__name__}: frame_idx={current_idx}, from={self.__file.name}"