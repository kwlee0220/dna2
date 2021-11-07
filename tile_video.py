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


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a localpath on the screen")
    parser.add_argument("file", nargs='+', metavar='file', help="target video file")
    parser.add_argument("--output_video", metavar="file", help="output video file", required=False)
    return parser.parse_known_args()

def capture(caps):
    frames = []
    for cap in caps:
        if not cap.is_open():
            return None
        _, _, frame = cap.capture()
        if frame is None:
            return None
        frames.append(frame)
    return frames

def merge(frames, size):
    frames = list([cv2.resize(frame, size, interpolation=cv2.INTER_AREA) for frame in frames])
    return np.vstack([np.hstack(frames[0:2]), np.hstack(frames[2:])])

if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    size = None
    caps = [VideoFileCapture(file, sync=False) for file in args.file]
    for cap in caps:
        cap.open()
        if size is None:
            size = (cap.size / 2).to_int().as_tuple()

    output = Path(args.output_video)
    ext = output.suffix.lower()
    if ext == '.mp4':
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    elif ext == '.avi':
        fourcc = cv2.VideoWriter_fourcc(*'DIVX')
    else:
        raise ValueError("unknown output video file extension: 'f{ext}'")
    fps = caps[0].fps
    writer = cv2.VideoWriter(str(output.resolve()), fourcc,
                                    fps, caps[0].size.as_tuple())

    while True:
        frames = capture(caps)
        if frames is None:
            break

        frame = merge(frames, size)
        writer.write(frame)

    writer.release()
    map(lambda c: cap.close(), caps)
    cv2.destroyAllWindows()