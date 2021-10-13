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
    __OS_OVERHEAD = 100 / 1000  # MAGIC NUMBER

    def __init__(self, capture, sync: bool=False, window_name: str=None,
                show_progress: bool=False, stop_at_the_last=False) -> None:
        self.capture = capture
        self.__sync = sync
        self.window_name = window_name
        self.show = window_name is not None
        self.show_progress = show_progress
        self.stop_at_the_last = stop_at_the_last

    # @abstractmethod
    def on_started(self) -> None:
        pass

    # @abstractmethod
    def on_stopped(self) -> None:
        pass

    # @abstractmethod
    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        return frame

    # @abstractmethod
    def set_control(self, key: int) -> int:
        return key

    def __enter__(self):
        self.capture.open()
        try:
            self.on_started()
        except Exception as e:
            self.capture.close()
            raise e

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        try:
            self.on_stopped()
        finally:
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
        capture_count = 0
        elapsed_avg = None
        fps_measured = 0
        overhead = 0
        fps = self.capture.fps
        show_fps = True
        sync_fps = self.__sync
        frame_interval = (1 - ImageProcessor.__OS_OVERHEAD) / fps

        if self.show_progress \
            and self.capture.frame_count is not None \
            and self.capture.frame_count > 1:
            progress = tqdm(total=self.capture.frame_count)
        else:
            progress = None

        key = ''
        wait_ts = 0
        while self.capture.is_open():
            started = time.time()
            ts, frame_idx, mat = self.capture.capture()
            if mat is None:
                break
            capture_count += 1

            mat = self.process_image(mat, frame_idx, ts)
            if self.window_name and self.show:
                if show_fps:
                    image = cv2.putText(mat, f'FPS={fps_measured:.2f}, frames={frame_idx}', (20, 20),
                                        cv2.FONT_HERSHEY_COMPLEX_SMALL, 1, color.RED, 2)
                cv2.imshow(self.window_name, mat)
            
            elapsed = (time.time() - started)
            if not self.__sync and not self.window_name:
                wait_millis = -1
            elif sync_fps:
                wait_millis = max((frame_interval - elapsed - overhead) * 1000, 1)
            else:
                wait_millis = 1 # 화면에 이미지를 출력하고, key input을 받아야 하므로 1로 세팅
                
            if wait_millis > 0:
                key = cv2.waitKey(int(wait_millis)) & 0xFF
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
            if capture_count > 10:
                elapsed_avg = elapsed_avg * (1-ImageProcessor.__ALPHA) + elapsed * ImageProcessor.__ALPHA
            elif capture_count > 1:
                elapsed_avg = elapsed_avg * 0.5 + elapsed * 0.5
            else:
                elapsed_avg = elapsed
            overhead = elapsed_avg - frame_interval
            fps_measured = 1 / elapsed_avg
            # print(f"fps={fps_measured:.3f}, overhead={overhead*1000:.0f}, wait_millis={wait_ts*1000:.0f}")

            if progress is not None:
                progress.update(1)

        if key != ord('q') and self.stop_at_the_last:
            cv2.waitKey(-1)

        cv2.destroyAllWindows()
        if progress:
            progress.close()

        return self.capture.frame_count