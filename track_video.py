from collections import defaultdict
from pathlib import Path

import numpy as np
import argparse
from omegaconf import OmegaConf

from dna import BBox
from dna.camera import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.track.track_callbacks import TrackWriter
from dna.platform import DNAPlatform


def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--conf", help="DNA framework configuration", default="conf/config.yaml")
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument("--camera_id", metavar="id", help="target camera id")
    group.add_argument("--input", metavar="source", help="input source")
    parser.add_argument("--begin_frame", type=int, help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, help="the last frame index", default=None)
    parser.add_argument("--sync", help="sync to fps", action="store_true")

    parser.add_argument("--tracker", help="tracker", default="tracker.deep_sort")
 
    parser.add_argument("--output", help="detection output file.", required=False)
    parser.add_argument("--output_video", help="output video file", required=False)
    parser.add_argument("--show", help="show detections.", action="store_true")
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    return parser.parse_known_args()

def parse_config_args(args):
    config_grp = defaultdict(list)
    for arg in args:
        idx = arg.find('=')
        if idx >= 0:
            key = arg[:idx]
            value = arg[idx+1:]
            idx = key.find('.')
            grp_key = key[:idx]
            key = key[idx+1:]
            config_grp[grp_key].append((key, value))
    return config_grp

def update_config(config, updates):
    for k, v in updates:
        OmegaConf.update(tracker_conf, k, v)


import dna.camera.utils as camera_utils
if __name__ == '__main__':
    args, unknown = parse_args()

    conf = OmegaConf.load(args.conf)
    config_grp = parse_config_args(unknown)

    cap = None
    domain = None
    blind_regions = None
    if args.input:
        cap = camera_utils.load_image_capture(args.input, sync=args.sync,
                                            begin_frame=args.begin_frame, end_frame=args.end_frame)
    else:
        dict = OmegaConf.to_container(conf.platform)

        platform = DNAPlatform.load(dict)
        _, camera_info = platform.get_resource("camera_infos", (args.camera_id,))

        cap = camera_utils.load_image_capture(camera_info.uri, size=camera_info.size, sync=args.sync,
                                            begin_frame=args.begin_frame, end_frame=args.end_frame)
        blind_regions = camera_info.blind_regions

    cap.open()
    sz = cap.size
    domain = BBox.from_tlbr(np.array([0, 0, sz.width, sz.height]))
    cap.close()

    tracker_conf = OmegaConf.select(conf, args.tracker)
    update_config(tracker_conf, config_grp.get('tracker', []))
    detector = DetectorLoader.load(tracker_conf.detector)
    tracker = DeepSORTTracker(detector, domain, tracker_conf, blind_regions=blind_regions)
    track_writer = TrackWriter(args.output) if args.output else None

    display_window_name = "output" if args.show else None
    with ObjectTrackingProcessor(cap, tracker, track_writer,
                                window_name=display_window_name, output_video=args.output_video,
                                show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )
