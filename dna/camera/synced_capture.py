import threading
from typing import Tuple
from datetime import datetime, timedelta
from pathlib import Path
from enum import Enum
from threading import Thread, Condition

import numpy as np
import cv2

from dna import Size2d
from .image_capture import ImageCapture
from .image_processor import ImageProcessor

class State(Enum):
    STOPPED = 1
    STARTING = 2
    RUNNING = 3
    STOPPING = 4

class CapturingProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture) -> None:
        super().__init__(capture)

        self.cap = capture
        self.state = State.STOPPED

        self.cond = Condition()
        self.frame = None
        self.frame_idx = -1
        self.ts = None

    def start(self) -> None:
        with self.cond:
            self.state = State.STARTING
            self.thread = Thread(target=self.run, args=tuple())
            self.thread.start()

            while self.state == State.STARTING:
                self.cond.wait()

    def stop(self) -> None:
        with self.cond:
            while self.state == State.STARTING:
                self.cond.wait()
            if self.state != State.RUNNING:
                return

            self.state = State.STOPPING
            self.cond.notifyAll()

            while self.state != State.STOPPED:
                self.cond.wait()
        self.thread.join()  

    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        with self.cond:
            if self.state == State.STOPPING:
                self.cap.close()
                self.state = State.STOPPED
            elif self.state == State.STARTING:
                self.state = State.RUNNING
            
            self.frame = frame
            self.frame_idx = frame_idx
            self.ts = ts
            self.cond.notifyAll()

        return frame

    def capture(self, min_frame_idx: int) -> Tuple[datetime, int, np.ndarray]:
        with self.cond:
            while True:
                while self.state == State.STARTING:
                    self.cond.wait()
                if self.state == State.RUNNING:
                    if self.frame_idx >= min_frame_idx:
                        return self.ts, self.frame_idx, self.frame
                    else:
                        self.cond.wait()
                else:
                    return None, None, None


class SyncImageCapture(ImageCapture):
    def __init__(self, capture: ImageCapture) -> None:
        self.__cap: ImageCapture = capture
        self.__proc = CapturingProcessor(capture)
        self.last_frame = -1

    def open(self) -> None:
        self.__cap.open()
        self.__proc.start()

    def close(self) -> None:
        self.__proc.stop()

    def is_open(self) -> bool:
        return self.__cap.is_open()

    @property
    def size(self) -> Size2d:
        return self.__cap.size

    @property
    def fps(self) -> int:
        return self.__cap.fps

    @property
    def frame_index(self) -> int:
        return self.last_frame

    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        return self.__proc.capture(self.last_frame + 1)