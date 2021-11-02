from typing import List
from pathlib import Path
from threading import Thread
import sys

from timeit import default_timer as timer
from datetime import timedelta
from dna.enhancer.world_transform import WorldTransform
from pubsub import PubSub, Queue
from omegaconf import OmegaConf
import numpy as np

from dna import Box, DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.camera import Camera
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, LogFileBasedObjectTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.track_event_uploader import TrackEventUploader
from dna.enhancer.local_path_uploader import LocalPathUploader
from dna.platform import DNAPlatform


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("node", nargs="?", metavar='node_id', help="target node_id")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    parser.add_argument("--track_file", help="Object track log file.", default=None)
    parser.add_argument("--tracker", help="tracker", default="tracker.deep_sort")

    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE, args.node)
    camera_info = Camera.from_conf(conf.camera)
    cap = camera_info.get_capture(sync=args.sync, begin_frame=args.begin_frame, end_frame=args.end_frame)

    if args.track_file is not None:
        tracker = LogFileBasedObjectTracker(args.track_file)
    else:
        detector = DetectorLoader.load(conf.tracker.detector)
        domain = Box.from_size(camera_info.size)
        tracker = DeepSORTTracker(detector, domain, conf.tracker, blind_regions=camera_info.blind_regions)

    platform = DNAPlatform.load_from_config(conf.platform)

    pubsub = PubSub()
    enhancer = TrackEventEnhancer(pubsub, camera_info.camera_id, conf.enhancer)

    wtrans = WorldTransform(camera_info.camera_id, pubsub, enhancer.subscribe(), conf.camera_geometry)
    thread = Thread(target=wtrans.run, args=tuple())
    thread.start()

    path_upload = LocalPathUploader(platform, wtrans.subscribe(), conf.enhancer.path_uploader)
    thread = Thread(target=path_upload.run, args=tuple())
    thread.start()

    te_upload = TrackEventUploader(platform, wtrans.subscribe(), conf.enhancer.event_uploader)
    thread = Thread(target=te_upload.run, args=tuple())
    thread.start()

    win_name = camera_info.camera_id if args.show else None
    with ObjectTrackingProcessor(cap, tracker, enhancer, window_name=win_name) as processor:
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )