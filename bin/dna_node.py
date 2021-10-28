from typing import List
from pathlib import Path
from threading import Thread
import sys

from timeit import default_timer as timer
from datetime import timedelta
from pubsub import PubSub, Queue
from omegaconf import OmegaConf
import numpy as np

import dna
import dna.camera as dna_cam
import dna.det as dna_det
from dna import Box
from dna.track import DeepSORTTracker, LogFileBasedObjectTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.track_event_uploader import TrackEventUploader
from dna.enhancer.local_path_uploader import LocalPathUploader
from dna.platform import DNAPlatform


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=dna.DNA_CONIFIG_FILE)
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    parser.add_argument("--track_file", help="Object track log file.", default=None)
    parser.add_argument("--tracker", help="tracker", default="tracker.deep_sort")

    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_known_args()

def update_config(config, updates):
    for k, v in updates:
        OmegaConf.update(tracker_conf, k, v)

if __name__ == '__main__':
    args, unknown = parse_args()

    conf = OmegaConf.load(args.conf)
    config_grp = dna.parse_config_args(unknown)

    platform = DNAPlatform.load_from_config(conf.platform)
    _, camera_info = platform.get_resource("camera_infos", (args.camera,))
    cap = dna_cam.load_image_capture(camera_info.uri, sync=args.sync, \
                                    begin_frame=args.begin_frame, end_frame=args.end_frame)

    if args.track_file is not None:
        tracker = LogFileBasedObjectTracker(args.track_file)
    else:
        tracker_conf = OmegaConf.select(conf, args.tracker)
        update_config(tracker_conf, config_grp.get('tracker', []))
        detector = dna_det.DetectorLoader.load(tracker_conf.detector)

        domain = Box.from_tlbr(np.array([0, 0, camera_info.size.width, camera_info.size.height]))
        tracker = DeepSORTTracker(detector, domain, tracker_conf, blind_regions=camera_info.blind_regions)

    pubsub = PubSub()
    enhancer_conf = OmegaConf.select(conf, "enhancer")
    enhancer = TrackEventEnhancer(pubsub, args.camera, enhancer_conf)

    ev_uploader_conf = OmegaConf.select(conf, "event_uploader")
    te_upload = TrackEventUploader(platform, enhancer.subscribe(), ev_uploader_conf)
    thread = Thread(target=te_upload.run, args=tuple())
    thread.start()

    path_uploader_conf = OmegaConf.select(conf, "path_uploader")
    path_upload = LocalPathUploader(platform, enhancer.subscribe(), path_uploader_conf)
    thread = Thread(target=path_upload.run, args=tuple())
    thread.start()

    win_name = args.camera if args.show else None
    with ObjectTrackingProcessor(cap, tracker, enhancer, window_name=win_name) as processor:
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )