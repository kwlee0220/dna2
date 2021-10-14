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
from .image_capture import ImageCapture


_ALPHA = 0.05
class ImageProcessor(metaclass=ABCMeta):

    def __init__(self, capture: ImageCapture, window_name: str=None,
                show_progress: bool=False, stop_at_the_last=False) -> None:
        self.__cap = capture
        self.window_name = window_name
        self.show = window_name is not None
        self.show_progress = show_progress
        self.stop_at_the_last = stop_at_the_last

    def on_started(self) -> None:
        pass

    def on_stopped(self) -> None:
        pass
 
    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        return frame

    def set_control(self, key: int) -> int:
        return key

    def __enter__(self):
        self.__cap.open()
        try:
            self.on_started()
        except Exception as e:
            self.__cap.close()
            raise e

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            self.on_stopped()
        finally:
            self.__cap.close()

    def run(self) -> int:
        capture_count = 0
        elapsed_avg = None
        fps_measured = 0
        show_fps = True

        if self.show_progress \
            and self.__cap.frame_count is not None \
            and self.__cap.frame_count > 1:
            progress = tqdm(total=self.__cap.frame_count)
        else:
            progress = None

        key = ''
        while self.__cap.is_open():
            started = time.time()
            ts, frame_idx, frame = self.__cap.capture()
            if frame is None:
                break
            capture_count += 1

            frame = self.process_image(frame, frame_idx, ts)
            if self.window_name and self.show:
                if show_fps:
                    image = cv2.putText(frame, f'FPS={fps_measured:.2f}, frames={frame_idx}', (20, 20),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
                cv2.imshow(self.window_name, frame)

                key = cv2.waitKey(int(1)) & 0xFF
                if key == ord('q'):
                    self.__cap.close()
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
            if progress is not None:
                progress.update(1)
            
            if capture_count > 10:
                elapsed_avg = elapsed_avg * (1-_ALPHA) + elapsed * _ALPHA
            elif capture_count > 1:
                elapsed_avg = elapsed_avg * 0.5 + elapsed * 0.5
            else:
                elapsed_avg = elapsed
            fps_measured = 1 / elapsed_avg

        if key != ord('q') and self.window_name and self.show and self.stop_at_the_last:
            cv2.waitKey(-1)

        if progress:
            progress.close()
        if self.window_name:
            cv2.destroyWindow(self.window_name)

        return self.__cap.frame_count