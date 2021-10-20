from collections import defaultdict
from pathlib import Path

import numpy as np
import argparse
from omegaconf import OmegaConf

from dna import BBox, utils as dna_utils, DNA_CONIFIG_FILE, parse_config_args
from dna.camera import ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.track.track_callbacks import TrackWriter
from dna.platform import DNAPlatform


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    parser.add_argument("--tracker", help="tracker", default="tracker.deep_sort")

    parser.add_argument("--output", metavar="file",
                        help="output detection file.", required=False)
    parser.add_argument("--output_video", metavar="file",
                        help="output video file", required=False)
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_known_args()

def update_config(config, updates):
    for k, v in updates:
        OmegaConf.update(tracker_conf, k, v)


if __name__ == '__main__':
    args, unknown = parse_args()

    conf = OmegaConf.load(args.conf)
    config_grp = parse_config_args(unknown)

    uri = args.camera
    cap_type = image_capture_type(uri)
    if cap_type == ImageCaptureType.PLATFORM:
        platform = DNAPlatform.load_from_config(conf.platform)
        _, camera_info = platform.get_resource("camera_infos", (uri,))
        uri = camera_info.uri
        blind_regions = camera_info.blind_regions
    else:
        blind_regions = None
    cap = load_image_capture(uri, sync=args.sync, begin_frame=args.begin_frame, end_frame=args.end_frame)

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
