from pathlib import Path

import numpy as np

from dna.camera import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.track.track_callbacks import TrackWriter


import argparse

from dna.types import BBox
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--max_age", type=int, help="max. # of frames to delete", default=20)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--output", help="detection output file.", required=False)
    parser.add_argument("--show", help="show detections.", action="store_true")
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    capture = VideoFileCapture(Path(args.input), sync=False)
    capture.open()
    sz = capture.size
    domain = BBox.from_tlbr(np.array([0, 0, sz.width-1, sz.height-1]))
    capture.close()

    detector = DetectorLoader.load(args.detector)

    det_dict = {'car': 'car', 'bus': 'bus', 'truck':'car'}

    dna_home_dir = Path(args.home)
    model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model640.pt'
    tracker = DeepSORTTracker(domain, detector, weights_file=model_file.absolute(),
                                matching_threshold=args.match_score,
                                max_iou_distance=args.max_iou_distance,
                                max_age=args.max_age,
                                det_dict = det_dict)
    track_writer = TrackWriter(args.output) if args.output else None

    display_window_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, tracker, track_writer,
                                window_name=display_window_name,
                                show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )