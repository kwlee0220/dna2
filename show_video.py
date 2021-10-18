from pathlib import Path
from datetime import datetime
import sys

import cv2
import numpy as np
import hydra

from dna import plot_utils
from dna.camera import VideoFileCapture, ImageProcessor
from dna.platform import DNAPlatform
from omegaconf import OmegaConf


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
    parser.add_argument("--conf", help="DNA framework configuration", default="conf/config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera_id", metavar="id", help="target camera id")
    group.add_argument("--input", metavar="source", help="input source")
    # parser.add_argument("--resize_ratio", type=float, help="image resizing ratio", default=None)
    # parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    # parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cap = None
    if args.input:
        from dna.camera.utils import load_image_capture
        cap = load_image_capture(args.input)
    else:
        conf = OmegaConf.load(args.conf)
        dict = OmegaConf.to_container(conf.platform)

        platform = DNAPlatform.load(dict)
        cap = platform.load_image_capture(args.camera_id)

    dna_home_dir = Path(args.home)
    with ImageProcessor(cap, window_name="output") as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={processor.fps_measured:.1f}" )