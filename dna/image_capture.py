from typing import Tuple
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
import time
from pathlib import Path

import numpy as np
import cv2

from dna import Size2i
import dna.utils as utils

class ImageCapture(metaclass=ABCMeta):
    @abstractmethod
    def is_open(self) -> bool:
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        pass

    @abstractmethod
    def size(self) -> Size2i:
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        pass

    def __enter__(self):
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.is_open():
            self.close()

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: frames={self.frame_index}/{self.frame_count}, "
                f"fps={self.fps:.0f}/s")


import sys
class VideoFileCapture(ImageCapture):
    def __init__(self, file: Path, begin_frame: int=1, end_frame: int=None) -> None:
        self.__file = file
        self.__vid = None     # None on if it is closed
        self.__fps = None
        self.__frame_count = -1
        self.__frame_index = -1

        # end_frame = end_frame if end_frame else sys.maxsize*2 + 1
        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.__frame_begin}, end={self.__frame_end}"))
        self.__frame_begin = begin_frame
        self.__frame_end = end_frame

        self.__size = None

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
        self.__size = Size2i([width, height])

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
            self.__frame_index += 1

        return datetime.now(), self.__frame_index, mat

    def __repr__(self) -> str:
        repr = super().__repr__()
        return repr + f", file={self.file}"


from threading import Condition
from .image_processor import ImageProcessor
class ImageCaptureLoop(ImageProcessor):
    def __init__(self, capture, cond: Condition) -> None:
        super().__init__(capture, window_name=None, show_progress=False)
        self.cond = cond

    def capture(self, target_frame_idx: int) -> Tuple[datetime, int, np.ndarray]:
        with self.cond:
            while self.frame_idx < target_frame_idx:
                self.cond.wait()
            return self.frame_ts, self.frame_idx, self.frame

    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        with self.cond:
            self.frame = frame
            self.frame_idx = frame_idx
            self.frame_ts = ts
            self.cond.notifyAll()

        return frame

    def on_started(self) -> None:
        pass

    def on_stopped(self) -> None:
        with self.cond:
            self.frame_idx = -1

    def set_control(self, key: int) -> int:
        return key

