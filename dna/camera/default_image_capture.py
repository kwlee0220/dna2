from typing import Tuple
import time

import numpy as np
import cv2

from dna import Size2d
from .image_capture import ImageCapture

_INIT_SIZE = Size2d(-1, -1)


class DefaultImageCapture(ImageCapture):
    def __init__(self, uri:str, target_size :Size2d=None, begin_frame: int=1, end_frame: int=None) -> None:
        """Create a DefaultImageCapture object.

        Args:
            uri (str): Resource identifier to the ImageCapture.
            target_size (Size2d, optional): Output image size. Defaults to None.
        """
        self.uri = uri
        self.__cap = None     # None on if it is closed
        self.__fps = -1
        self.__size = target_size if target_size else _INIT_SIZE
        self.interpolation = None
        self.__frame_index = -1

        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.begin_frame}, end={self.end_frame}"))
        self.begin_frame = begin_frame
        self.end_frame = end_frame

    def is_open(self) -> bool:
        return self.__cap is not None

    def open(self) -> None:
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")

        self.__cap = cv2.VideoCapture(self.uri)
        if not self.__cap.isOpened():
            self.__cap = None
            raise IOError(f"fails to open video capture: '{self.uri}'")

        self.__frame_index = 0
        self.__fps = self.__cap.get(cv2.CAP_PROP_FPS)

        width = int(self.__cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(self.__cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        src_size = Size2d(width, height)
        if self.__size == _INIT_SIZE:
            self.__size = src_size
        elif self.__size.area() < src_size.area():
            self.interpolation = cv2.INTER_AREA
        elif self.__size.area() > src_size.area():
            self.interpolation = cv2.INTER_LINEAR
        else:
            self.__size = src_size

        while self.__frame_index < self.begin_frame-1:
            _, mat = self.__cap.read()
            if mat:
                self.__frame_index += 1

    def close(self) -> None:
        if self.__cap:
            self.__cap.release()
            self.__cap = None

    @property
    def video_capture(self):
        return self.__cap

    @property
    def size(self) -> Size2d:
        return self.__size

    @property
    def fps(self) -> int:
        return self.__fps

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    def capture(self) -> Tuple[float, int, np.ndarray]:
        if not self.is_open():
            raise ValueError(f"{self.__class__.__name__}: not opened")
        if self.end_frame and self.__frame_index >= self.end_frame:
            return time.time(), self.__frame_index, None

        _, mat = self.__cap.read()
        if mat is not None:
            if self.interpolation:
                mat = cv2.resize(mat, self.size.as_tuple(), interpolation=self.interpolation)
            self.__frame_index += 1

        return time.time(), self.__frame_index, mat

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        end_frame = self.end_frame if self.end_frame else ""
        return (f"{__class__.__name__}[{state}]: uri={self.uri}[{self.begin_frame}:{end_frame}], "
                f"size={self.__size}, frames={self.__frame_index}, fps={self.__fps:.0f}/s")