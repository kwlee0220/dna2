# import sys
from pathlib import Path
from abc import ABCMeta, abstractmethod
from contextlib import suppress
import time

from tqdm import tqdm
import numpy as np
import cv2

import dna
from dna import color
from .image_capture import ImageCapture


_ALPHA = 0.05
class ImageProcessor(metaclass=ABCMeta):
    def __init__(self, capture: ImageCapture, window_name: str=None, output_video: Path=None,
                show_progress: bool=False, stop_at_the_last=False) -> None:
        self.__cap = capture
        self.window_name = window_name
        self.output_video = output_video if not output_video or isinstance(output_video, Path) \
                                        else Path(output_video)
        self.writer = None
        self.show_progress = show_progress
        self.stop_at_the_last = stop_at_the_last
        self.__fps_measured = -1

    @property
    def fps_measured(self) -> float:
        return self.__fps_measured

    def on_started(self, capture: ImageCapture) -> None:
        pass

    def on_stopped(self) -> None:
        pass
 
    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        return frame

    def set_control(self, key: int) -> int:
        return key

    def __enter__(self):
        self.__cap.open()
        try:
            self.on_started(self.__cap)
            if self.window_name:
                cv2.namedWindow(self.window_name)
            if self.output_video:
                fourcc = None
                ext = self.output_video.suffix.lower()
                if ext == '.mp4':
                    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                elif ext == '.avi':
                    fourcc = cv2.VideoWriter_fourcc(*'DIVX')
                else:
                    raise ValueError("unknown output video file extension: 'f{ext}'")
                self.writer = cv2.VideoWriter(str(self.output_video.resolve()), fourcc,
                                                self.__cap.fps, self.__cap.size.as_tuple())
        except Exception as e:
            if self.writer:
                self.writer.release()
            self.__cap.close()
            raise e

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            self.on_stopped()
        finally:
            if self.window_name:
                cv2.destroyWindow(self.window_name)
            self.__cap.close()

    def run(self) -> int:
        capture_count = 0
        elapsed_avg = None
        self.__fps_measured = 0

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

            dna.DEBUG_FRAME_IDX = frame_idx
            frame = self.process_image(frame, frame_idx, ts)
            frame = cv2.putText(frame, f'frames={frame_idx}, fps={self.__fps_measured:.2f}', (10, 20),
                            cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
            if self.writer:
                self.writer.write(frame)
            if dna.DEBUG_PRINT_COST:
                print("---------------------------------------------------------------------")
                
            if self.window_name:
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

            elapsed = time.time() - started
            if progress is not None:
                progress.update(1)
            
            if capture_count > 10:
                elapsed_avg = elapsed_avg * (1-_ALPHA) + elapsed * _ALPHA
            elif capture_count > 1:
                elapsed_avg = elapsed_avg * 0.5 + elapsed * 0.5
            else:
                elapsed_avg = elapsed
            self.__fps_measured = 1 / elapsed_avg

        if key != ord('q') and self.window_name and self.stop_at_the_last:
            cv2.waitKey(-1)

        if progress:
            progress.close()

        return capture_count