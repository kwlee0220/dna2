from typing import Tuple
from abc import ABCMeta, abstractmethod
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import cv2

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


class VideoFileCapture(ImageCapture):
    def __init__(self, file: Path, start_ts: datetime =None, fps: float=-1) -> None:
        self.__file = file
        self.start_ts = datetime.now()
        self.__fps = fps if fps > 0 else -1
        self.offset = (start_ts - self.start_ts) if start_ts else timedelta(milliseconds=0)
        self.__vid = None     # None on if it is closed
        self.__frame_count = -1
        self.__frame_index = -1

    def is_open(self) -> bool:
        return self.__vid is not None

    def open(self) -> None:
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")

        self.__vid = cv2.VideoCapture(str(self.__file))
        if not self.__vid.isOpened():
            self.__vid = None
            raise IOError(f"fails to open video capture: '{self.target}'")

        self.__frame_count = self.__vid.get(cv2.CAP_PROP_FRAME_COUNT)
        if self.__fps < 0:
            self.__fps = self.__vid.get(cv2.CAP_PROP_FPS)
        self.__frame_index = 0

    def close(self) -> None:
        if self.__vid:
            self.__vid.release()
            self.__vid = None

    @property
    def file(self) -> Path:
        return self.__file

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

        _, mat = self.__vid.read()
        if mat is not None:
            self.__frame_index += 1
        return datetime.now() + self.offset, self.__frame_index, mat

    def __repr__(self) -> str:
        repr = super().__repr__()
        return repr + f", file={self.file}"