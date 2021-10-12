from typing import List
from pathlib import Path
from threading import Thread
import sys

import numpy as np
from pubsub import PubSub, Queue
import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, LogFileBasedObjectTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.types import TrackEvent
from dna.enhancer.track_event_uploader import TrackEventUploader
from dna.enhancer.trajectory_uploader import TrajectoryUploader
from dna.platform import DNAPlatform
import dna.utils as utils


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--camera_id", help="camera id")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--track_file", help="Object track log file.", default=None)
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--max_age", type=int, help="max. # of frames to delete", default=20)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--show", help="show detections.", action="store_true")

    parser.add_argument("--db_host", help="host name of DNA data platform", default="localhost")
    parser.add_argument("--db_port", type=int, help="port number of DNA data platform", default=5432)
    parser.add_argument("--db_name", help="database name", default="dna")
    parser.add_argument("--db_user", help="user name", default="postgres")
    parser.add_argument("--db_passwd", help="password", default="dna2021")
    return parser.parse_args()


import dna.utils as utils

if __name__ == '__main__':
    args = parse_args()

    platform = DNAPlatform(host=args.db_host, port=args.db_port,
                            user=args.db_user, password=args.db_passwd, dbname=args.db_name)
    conn = platform.connect()

    camera_info_set = platform.get_resource_set("camera_infos")
    camera_info = camera_info_set.get((args.camera_id,))
    if camera_info is None:
        print(f"unknown camera_id: '{args.camera_id}'", file=sys.stderr)
        exit(-1)

    pubsub = PubSub()
    enhancer = TrackEventEnhancer(pubsub, args.camera_id)

    if args.track_file is not None:
        tracker = LogFileBasedObjectTracker(args.track_file)
    else:
        dna_home_dir = Path(args.home)
        detector = DetectorLoader.load(args.detector)
        model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model640.pt'
        tracker = DeepSORTTracker(detector, weights_file=model_file.absolute(),
                                    matching_threshold=args.match_score,
                                    max_iou_distance=args.max_iou_distance,
                                    max_age=args.max_age)

    te_upload = TrackEventUploader(platform, enhancer.subscribe())
    thread = Thread(target=te_upload.run, args=tuple())
    thread.start()

    trj_upload = TrajectoryUploader(platform, enhancer.subscribe())
    thread = Thread(target=trj_upload.run, args=tuple())
    thread.start()

    capture = VideoFileCapture(Path(args.input))
    win_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, tracker, enhancer, window_name=win_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )