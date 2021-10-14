from typing import Tuple
import time
from pathlib import Path

import numpy as np
import cv2

from dna import Size2i
from .default_image_capture import DefaultImageCapture


_ALPHA = 0.3
class VideoFileCapture(DefaultImageCapture):
    def __init__(self, file: Path, sync :bool=True, target_size :Size2i=None,
                begin_frame: int=1, end_frame: int=None) -> None:
        """Create a VideoFile ImageCapture object.

        Args:
            file (Path): A video file path
            sync (bool, optional): Sync to the fps. Defaults to False.
            target_size (Size2i, optional): Output image size. Defaults to None.
            begin_frame (int, optional): The index of the first frame. Defaults to 1.
            end_frame (int, optional): The index of the last frame. Defaults to None.

        Raises:
            ValueError: 'begin_frame' or 'end_frame' are invalid.
        """
        super().__init__(str(file.resolve()), target_size=target_size)
        self.file = file
        self.sync = sync
        self.__frame_count = -1

        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.begin_frame}, end={self.end_frame}"))
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.delta = 0

    def open(self) -> None:
        super().open()

        if not self.end_frame:
            self.end_frame = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_count = self.end_frame - self.begin_frame + 1
        self.__fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.interval = 1 / self.__fps

        if self.begin_frame > 1:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame)
        self.__frame_index = self.begin_frame-1

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    def capture(self) -> Tuple[float, int, np.ndarray]:
        started = time.time()

        if self.frame_index >= self.end_frame:
            return None, -1, None

        ts, frame_idx, frame = super().capture()
        if self.sync:
            remains = self.interval - (ts - started) - self.delta
            if remains > 0:
                time.sleep(remains)

                ts = time.time()
                delta = (ts - started) - self.interval
                if self.delta == 0: # for the initial frame
                    self.delta = delta
                else:
                    self.delta += (_ALPHA * delta)

        return ts, frame_idx, frame


    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: uri={self.uri}, size={self.size}, "
                f"frames={self.frame_index}/{self.__frame_count}, fps={self.fps:.0f}/s")

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