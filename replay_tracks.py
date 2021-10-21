from pathlib import Path
import numpy as np

from dna import color
from dna.track import LogFileBasedObjectTracker, ObjectTrackingProcessor
from dna.camera import load_image_capture
from dna.platform import DNAPlatform
from omegaconf import OmegaConf


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generating tracking events from a track-file")
    parser.add_argument("--conf", help="DNA framework configuration", default="conf/config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera_id", metavar="id", help="target camera id")
    group.add_argument("--input", metavar="source", help="input source")
    parser.add_argument("--track_file", help="Object detection algorithm.")
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cap = None
    if args.input:
        cap = load_image_capture(args.input)
    else:
        conf = OmegaConf.load(args.conf)
        dict = OmegaConf.to_container(conf.platform)

        platform = DNAPlatform.load(dict)
        cap = platform.load_image_capture(args.camera_id)

    tracker = LogFileBasedObjectTracker(args.track_file)

    win_name = "output" if args.show else None
    with ObjectTrackingProcessor(cap, tracker, window_name=win_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )