from pathlib import Path
import numpy as np

from dna import  VideoFileCapture
from dna import color
from dna.track import LogFileBasedObjectTracker, ObjectTrackingProcessor


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generating tracking events from a track-file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--track_file", help="Object detection algorithm.")
    parser.add_argument("--video_file", help="input source.", required=True)
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # open target video file temporarily to find fps, which will be used
    # in calculating 'max_age'
    capture = VideoFileCapture(Path(args.video_file))

    dna_home_dir = Path(args.home)
    tracker = LogFileBasedObjectTracker(args.track_file)

    win_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, tracker, window_name=win_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )