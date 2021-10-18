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
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--max_age", type=int, help="max. # of frames to delete", default=20)

    parser.add_argument("--output", help="detection output file.", required=False)
    parser.add_argument("--output_video", help="output video file", required=False)
    parser.add_argument("--show", help="show detections.", action="store_true")
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    cap = None
    domain = None
    blind_regions = None
    if args.input:
        from dna.camera.utils import load_image_capture
        cap = load_image_capture(args.input, sync=args.sync)
    else:
        conf = OmegaConf.load(args.conf)
        dict = OmegaConf.to_container(conf.platform)

        platform = DNAPlatform.load(dict)
        camera_info = platform.get_resource("camera_infos", (args.camera_id,))

        import dna.camera.utils as camera_utils
        cap = camera_utils.load_image_capture(camera_info.uri, size=camera_info.size, sync=args.sync)
        blind_regions = camera_info.blind_regions

    cap.open()
    sz = cap.size
    domain = BBox.from_tlbr(np.array([0, 0, sz.width, sz.height]))
    cap.close()

    detector = DetectorLoader.load(args.detector)
    tracker_conf = conf.tracker.deep_sort
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