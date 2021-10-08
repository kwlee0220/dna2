# import sys
from pathlib import Path
from abc import ABCMeta, abstractmethod
from contextlib import suppress
import time
from datetime import datetime

from tqdm import tqdm
import numpy as np
import cv2

from dna import color


class ImageProcessor(metaclass=ABCMeta):
    __ALPHA = 0.05

    def __init__(self, capture, window_name: str=None, show_progress: bool=False) -> None:
        self.capture = capture
        self.window_name = window_name
        self.show = window_name is not None
        self.show_progress = show_progress

    @abstractmethod
    def on_started(self) -> None:
        pass

    @abstractmethod
    def on_stopped(self) -> None:
        pass

    @abstractmethod
    def process_image(self, utc_epoch: int, frame_idx: int, mat: np.ndarray) -> np.ndarray:
        pass

    @abstractmethod
    def set_control(self, key: int) -> int:
        pass

    def __enter__(self):
        self.capture.open()
        try:
            self.on_started()
        except Exception as e:
            self.capture.close()
            raise e

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        with suppress(Exception): self.on_stopped()
        self.capture.close()

    @property
    def frame_count(self) -> int:
        return self.capture.frame_count

    @property
    def fps(self) -> int:
        return self.capture.fps

    @property
    def frame_index(self) -> int:
        return self.capture.frame_index

    def run(self) -> int:
        elapsed_avg = None
        show_fps = True
        sync_fps = True
        fps = 0

        video_interval = 1000 / self.capture.fps
        if self.show_progress and self.capture.frame_count is not None and self.capture.frame_count > 1:
            progress = tqdm(total=self.capture.frame_count)
        else:
            progress = None

        while self.capture.is_open():
            started = time.time()
            ts, frame_idx, mat = self.capture.capture()
            if mat is None:
                break

            mat = self.process_image(ts, frame_idx, mat)
            wait_millis = 1
            if self.window_name and self.show:
                if show_fps:
                    image = cv2.putText(mat, f'FPS={fps:.2f}, frames={frame_idx}', (20, 20),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
                cv2.imshow(self.window_name, mat)

                if  sync_fps:
                    wait_millis = int(video_interval - (time.time() - started)*1000) - 7
                    wait_millis = max(1, wait_millis)
                key = cv2.waitKey(wait_millis) & 0xFF
                if key == ord('q'):
                    self.capture.close()
                    break
                elif key == ord(' '):
                    while True:
                        key = cv2.waitKey(1000 * 60 * 60) & 0xFF
                        if key == ord(' '):
                            break
                else:
                    key = self.set_control(key)
                    if key == ord('v'):
                        self.show = not self.show
                    elif key == ord('f'):
                        show_fps = not show_fps
                    elif key == ord('y'):
                        sync_fps = not sync_fps
            
            elapsed = time.time() - started
            if frame_idx > 10:
                elapsed_avg = elapsed_avg * (1-ImageProcessor.__ALPHA) + elapsed * ImageProcessor.__ALPHA
            elif frame_idx > 1:
                elapsed_avg = elapsed_avg * 0.5 + elapsed * 0.5
            else:
                elapsed_avg = elapsed
            fps = 1 / elapsed_avg

            if progress is not None:
                progress.update(1)
            # print('-----------------------------------------------------------------')

        cv2.destroyAllWindows()
        if progress:
            progress.close()

        return self.capture.frame_count