from typing import Tuple
from datetime import datetime, timedelta
from pathlib import Path
import sys

import numpy as np
import cv2

from dna import Size2i
from .image_capture import ImageCapture

_INIT_SIZE = Size2i(-1, -1)


class DefaultImageCapture(ImageCapture):
    def __init__(self, uri:str, target_size :Size2i=None) -> None:
        self.uri = uri
        self.cap = None     # None on if it is closed
        self.__fps = -1
        self.__size = target_size if target_size else _INIT_SIZE
        self.interpolation = None
        self.__frame_index = -1

    def is_open(self) -> bool:
        return self.cap is not None

    def open(self) -> None:
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")

        self.cap = cv2.VideoCapture(self.uri)
        if not self.cap.isOpened():
            self.cap = None
            raise IOError(f"fails to open video capture: '{self.uri}'")

        self.__frame_index = 0
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)

        width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_size = Size2i(width, height)
        if self.__size == _INIT_SIZE:
            self.__size = src_size
        elif self.__size.area() < src_size.area():
            self.interpolation = cv2.INTER_AREA
        elif self.__size.area > src_size.area:
            self.interpolation = cv2.INTER_LINEAR
        else:
            self.__size = src_size

    def close(self) -> None:
        if self.cap:
            self.cap.release()
            self.cap = None

    @property
    def size(self) -> Size2i:
        return self.__size

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        if not self.is_open():
            raise ValueError(f"{self.__class__.__name__}: not opened")

        _, mat = self.cap.read()
        if mat is not None:
            if self.interpolation:
                mat = cv2.resize(mat, self.size.as_tuple(), interpolation=self.interpolation)
            self.__frame_index += 1

        return datetime.now(), self.__frame_index, mat

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: uri={self.uri}, "
                f"size={self.__size}, frames={self.__frame_index}, fps={self.__fps:.0f}/s")