from typing import Tuple
import time

import numpy as np
import cv2

from dna import Size2d
from .image_capture import ImageCapture

_INIT_SIZE = Size2d(-1, -1)


class VideoCaptureHandle:
    def __init__(self, uri:str, target_size :Size2d=None) -> None:
        self.uri = uri
        self.cap = None     # None on if it is closed
        self.fps = -1
        self.size = target_size if target_size else _INIT_SIZE
        self.interpolation = None
        self.frame_index = -1

    def is_open(self) -> bool:
        return self.cap is not None

    def open(self) -> Tuple[cv2.VideoCapture, Size2d, int]:
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")

        self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            self.cap = None
            raise IOError(f"fails to open video capture: '{self.uri}'")

        self.frame_index = 0
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_size = Size2d(width, height)
        if self.size == _INIT_SIZE:
            self.size = src_size
        elif self.size.area() < src_size.area():
            self.interpolation = cv2.INTER_AREA
        elif self.size.area() > src_size.area():
            self.interpolation = cv2.INTER_LINEAR
        else:
            self.size = src_size

        return self.cap, self.size, self.fps

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    def capture(self) -> Tuple[float, int, np.ndarray]:
        if not self.is_open():
            raise ValueError(f"{self.__class__.__name__}: not opened")

        _, mat = self.cap.read()
        if mat is not None:
            if self.interpolation:
                mat = cv2.resize(mat, self.size.as_tuple(), interpolation=self.interpolation)
            self.frame_index += 1

        return time.time(), self.frame_index, mat

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: uri={self.uri}, "
                f"size={self.size}, frames={self.frame_index}, fps={self.fps:.0f}/s")