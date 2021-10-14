from pathlib import Path
from datetime import datetime
import sys

import cv2
import numpy as np

from dna import plot_utils
from dna.camera import VideoFileCapture, ImageProcessor


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
    parser.add_argument("--resize_ratio", type=float, help="image resizing ratio", default=None)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--begin_frame", type=int, help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, help="the last frame index", default=None)
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    target_size = None
    if args.resize_ratio:
        size, fps = VideoFileCapture.load_camera_info(Path(args.input))
        target_size = size * args.resize_ratio
    cap = VideoFileCapture(Path(args.input), target_size=target_size,
                            begin_frame=args.begin_frame, end_frame=args.end_frame)

    dna_home_dir = Path(args.home)
    with ImageProcessor(cap, window_name="output") as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )