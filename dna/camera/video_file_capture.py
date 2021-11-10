from typing import Tuple
import time
from pathlib import Path

import numpy as np
import cv2

from dna import Size2d
from dna.camera.image_capture import ImageCapture
from dna.camera.video_capture_handle import VideoCaptureHandle


_ALPHA = 0.3
class VideoFileCapture(ImageCapture):
    def __init__(self, file: Path, sync :bool=True, target_size :Size2d=None,
                begin_frame: int=1, end_frame: int=None) -> None:
        """Create a VideoFile ImageCapture object.

        Args:
            file (Path): A video file path
            sync (bool, optional): Sync to the fps. Defaults to False.
            target_size (Size2d, optional): Output image size. Defaults to None.
            begin_frame (int, optional): The index of the first frame. Defaults to 1.
            end_frame (int, optional): The index of the last frame. Defaults to None.

        Raises:
            ValueError: 'begin_frame' or 'end_frame' are invalid.
        """
        if isinstance(file, Path):
            file = str(file.resolve())
        else:
            file = str(file)
        self.__vch = VideoCaptureHandle(file, target_size=target_size)
        self.file = file
        self.sync = sync
        self.__frame_index = -1
        self.__frame_count = -1

        if begin_frame <= 0 or (end_frame and (end_frame < begin_frame)):
            raise ValueError((f"invalid [begin,end] frame range: "
                                f"begin={self.begin_frame}, end={self.end_frame}"))
        self.begin_frame = begin_frame
        self.end_frame = end_frame
        self.delta = 0

    def is_open(self) -> bool:
        return self.__vch.is_open()

    def open(self) -> None:
        cap, _, fps = self.__vch.open()

        if not self.end_frame:
            self.end_frame = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.__frame_count = self.end_frame - self.begin_frame + 1
        self.interval = 1 / fps

        if self.begin_frame > 1:
            cap.set(cv2.CAP_PROP_POS_FRAMES, self.begin_frame-1)
        self.__frame_index = self.begin_frame -1

    def close(self) -> None:
        self.__vch.close()

    @property
    def size(self) -> Size2d:
        return self.__vch.size

    @property
    def fps(self) -> int:
        return self.__vch.fps

    @property
    def frame_index(self) -> int:
        return self.__frame_index

    @property
    def frame_count(self) -> int:
        return self.__frame_count

    def capture(self) -> Tuple[float, int, np.ndarray]:
        if not self.is_open():
            raise ValueError(f"{self.__class__.__name__}: not opened")
        started = time.time()

        if self.frame_index >= self.end_frame:
            return None, -1, None

        ts, _, frame = self.__vch.capture()
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
        self.__frame_index += 1

        return ts, self.__frame_index, frame


    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[{state}]: uri={self.uri}, size={self.size}, "
                f"frames={self.frame_index}/{self.__frame_count}, fps={self.fps:.0f}/s")

    @staticmethod
    def load_camera_info(file: Path) -> Tuple[Size2d, int]:
        vch = VideoCaptureHandle(str(file.resolve()))
        cap, size, fps = vch.open()
        vch.close()

        return size, fps