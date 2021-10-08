from datetime import datetime
from typing import List
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from pubsub import PubSub

from dna import VideoFileCapture, BBox
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.types import TrackEvent

def store_track_event(event: TrackEvent):
    print(event)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--camera_id", help="camera id")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--max_age", type=int, help="max. # of frames to delete", default=30)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_args()


import dna.utils as utils

if __name__ == '__main__':
    args = parse_args()

    capture = VideoFileCapture(Path(args.input))
    detector = DetectorLoader.load(args.detector)

    dna_home_dir = Path(args.home)
    model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model640.pt'
    tracker = DeepSORTTracker(detector, weights_file=model_file.absolute(),
                                matching_threshold=args.match_score,
                                max_iou_distance=args.max_iou_distance,
                                max_age=args.max_age)

    pubsub = PubSub()
    enhancer = TrackEventEnhancer(pubsub, args.camera_id, store_track_event)

    win_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, tracker, enhancer, window_name=win_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )