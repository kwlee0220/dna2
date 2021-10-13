from typing import Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import cv2

from dna import Size2i
from .image_capture import ImageCapture

_INIT_SIZE = Size2i(-1, -1)

import sys
class VideoFileCapture(ImageCapture):
    def __init__(self, file: Path, target_size :Size2i=None,
                begin_frame: int=1, end_frame: int=None) -> None:
        self.__file = file
        self.__vid = None     # None on if it is closed
        self.__size = target_size if target_size else _INIT_SIZE
        self.interpolation = None
        self.__fps = -1
        self.__frame_count = -1
        self.__frame_index = -1

        # end_frame = end_frame if end_frame else sys.maxsize*2 + 1
        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.__frame_begin}, end={self.__frame_end}"))
        self.__frame_begin = begin_frame
        self.__frame_end = end_frame

    def is_open(self) -> bool:
        return self.__vid is not None

    def open(self) -> None:
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")

        self.__vid = cv2.VideoCapture(str(self.__file))
        if not self.__vid.isOpened():
            self.__vid = None
            raise IOError(f"fails to open video capture: '{self.__file}'")

        self.__frame_end = self.__frame_end if self.__frame_end else int(self.__vid.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_count = self.__frame_end - self.__frame_begin + 1
        self.__fps = self.__vid.get(cv2.CAP_PROP_FPS)

        if self.__frame_begin > 1:
            self.__vid.set(cv2.CAP_PROP_POS_FRAMES, self.__frame_begin)
        self.__frame_index = self.__frame_begin-1

        width = int(self.__vid.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__vid.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_size = Size2i(width, height)
        if self.__size == _INIT_SIZE:
            self.__size = src_size
        elif self.__size.area() < src_size.area():
            # self.interpolation = cv2.INTER_AREA
            self.interpolation = cv2.INTER_LINEAR
        elif self.__size.area > src_size.area:
            self.interpolation = cv2.INTER_LINEAR
        else:
            self.__size = src_size

    def close(self) -> None:
        if self.__vid:
            self.__vid.release()
            self.__vid = None

    @property
    def file(self) -> Path:
        return self.__file

    @property
    def size(self) -> Size2i:
        return self.__size

    @property
    def fps(self) -> float:
        return self.__fps

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        if not self.is_open():
            raise ValueError(f"{self.__class__.__name__}: not opened")

        if self.__frame_index >= self.__frame_end:
            return datetime.now(), -1, None

        _, mat = self.__vid.read()
        if mat is not None:
            if self.interpolation:
                mat = cv2.resize(mat, self.size.as_tuple(), interpolation=self.interpolation)
            self.__frame_index += 1

        return datetime.now(), self.__frame_index, mat

    @staticmethod
    def load_camera_info(file: Path) -> Tuple[Size2i, float]:
        cap = cv2.VideoCapture(str(file))
        if not cap.isOpened():
            raise IOError(f"fails to open video capture: '{file}'")

        try:
            fps = cap.get(cv2.CAP_PROP_FPS)
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            size = Size2i(width, height)
            
            return size, fps
        finally:
            cap.release()

    def __repr__(self) -> str:
        repr = super().__repr__()
        return repr + f", file={self.file}"