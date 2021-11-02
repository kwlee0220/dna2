from typing import List, Union
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from collections import namedtuple

from omegaconf import OmegaConf
import numpy as np
import cv2
from tqdm import tqdm

from dna import Box, DNA_CONIFIG_FILE, parse_config_args, load_config, color
from dna.camera import ImageCapture, ImageProcessor, VideoFileCapture
from dna.track import ObjectTracker, Track, LogFileBasedObjectTracker
from dna.platform import DNAPlatform

class FpsReducer(ImageProcessor):
    def __init__(self, capture: ImageCapture, output_video, skip:int) -> None:
        super().__init__(capture, show_progress=True)

        self.skip = skip
        self.output_video = output_video
        self.skip = skip

    def on_started(self, capture: ImageCapture) -> None:
        ext = self.output_video.suffix.lower()
        if ext == '.mp4':
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        elif ext == '.avi':
            fourcc = cv2.VideoWriter_fourcc(*'DIVX')
        else:
            raise ValueError("unknown output video file extension: 'f{ext}'")
        fps = capture.fps / self.skip
        self.writer = cv2.VideoWriter(str(self.output_video.resolve()), fourcc,
                                        fps, capture.size.as_tuple())

    def on_stopped(self) -> None:
        self.writer.release()

    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        if frame_idx % self.skip == 1:
            self.writer.write(frame)
        return frame


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a localpath on the screen")
    parser.add_argument("file", metavar='file', help="target video file")
    parser.add_argument("--skip", type=int, help="skip count")
    parser.add_argument("--output_video", metavar="file", help="output video file", required=False)
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)
    
    capture = VideoFileCapture(args.file, sync=False)
    capture.open()

    output = Path(args.output_video)
    ext = output.suffix.lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    else:
        raise ValueError("unknown output video file extension: 'f{ext}'")
    fps = capture.fps / args.skip
    writer = cv2.VideoWriter(str(output.resolve()), fourcc,
                                    fps, capture.size.as_tuple())

    progress = tqdm(total=capture.frame_count)
    while capture.is_open():
        ts, frame_idx, frame = capture.capture()
        if frame is None:
            break
        if frame_idx % args.skip == 1:
            writer.write(frame)
        progress.update(1)
    progress.close()
    capture.close()