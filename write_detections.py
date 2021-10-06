import sys
from pathlib import Path
import time
import cv2

from dna import plot_utils
from dna.det import DetectorLoader

_file_handle = None
def write_detection(frame_count, det_list):
    for det in det_list:
        msg = (f"{frame_count},-1,{det.bbox.tl.x},{det.bbox.tl.y},{det.bbox.width},"
                f"{det.bbox.height},{det.score:.3f},-1,-1,-1")
        if _file_handle:
            _file_handle.write(msg + '\n')
        else:
            print(msg)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", required=True)
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4_torch")
    parser.add_argument("--input", help="input source.", default="0")
    parser.add_argument("--output", help="output file.", required=False)
    parser.add_argument("--display", help="Display detection output.", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    DetectorLoader.dna_home_dir = Path(args.home)
    detector = DetectorLoader.load(args.detector)

    vid = cv2.VideoCapture(args.input)

    import detect_video
    if args.output:
        with open(args.output, 'w') as fhandle:
            _file_handle = fhandle
            detect_video.run(vid, detector, args.display, write_detection)
    else:
        detect_video.run(vid, detector, args.display, write_detection)