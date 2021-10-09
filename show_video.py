from pathlib import Path
from datetime import datetime

import cv2
import numpy as np

from dna import plot_utils, VideoFileCapture, ImageProcessor


class VideoFileDisplayer(ImageProcessor):
    def __init__(self, capture, window_name: str="video_output") -> None:
        super().__init__(capture, window_name=window_name, show_progress=False)

    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        return frame

    def on_started(self) -> None:
        pass

    def on_stopped(self) -> None:
        pass

    def set_control(self, key: int) -> int:
        return key


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Display a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--input", help="input source.", required=True)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    capture = VideoFileCapture(Path(args.input))
    dna_home_dir = Path(args.home)

    with VideoFileDisplayer(capture) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )