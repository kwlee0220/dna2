from typing import Tuple
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import cv2

from dna import Size2i
from .default_image_capture import DefaultImageCapture


class VideoFileCapture(DefaultImageCapture):
    def __init__(self, file: Path, target_size :Size2i=None,
                begin_frame: int=1, end_frame: int=None) -> None:
        super().__init__(str(file.resolve()), target_size=target_size)
        self.file = file
        self.frame_count = -1

        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.begin_frame}, end={self.end_frame}"))
        self.begin_frame = begin_frame
        self.end_frame = end_frame

    def open(self) -> None:
        super().open()

        self.end_frame = self.end_frame if self.end_frame else int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.frame_count = self.end_frame - self.begin_frame + 1
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)

        if self.begin_frame > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame)
        self.__frame_index = self.begin_frame-1

    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        if self.frame_index >= self.end_frame:
            return datetime.now(), -1, None

        print(self)
        return super().capture()

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: uri={self.uri}, size={self.size}, "
                f"frames={self.frame_index}/{self.frame_count}, fps={self.fps:.0f}/s")

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