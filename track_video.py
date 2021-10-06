from typing import List
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from dna import ImageProcessor, VideoFileCapture
from dna import plot_utils, color
from dna.det import Detection, DetectorLoader, ObjectDetector
from dna.track import ObjectTracker, Track, DeepSORTTracker, TrackerCallback

class ObjectTrackingProcessor(ImageProcessor):
    def __init__(self, capture, detector: ObjectDetector, tracker: ObjectTracker, callback: TrackerCallback,
                window_name:str=None, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, show_progress=show_progress)

        self.detector = detector
        self.tracker = tracker
        self.show_label = True
        self.callback = callback

    def on_started(self) -> None:
        if self.callback:
            self.callback.track_started()
        return self

    def on_stopped(self) -> None:
        if self.callback:
            self.callback.track_stopped()

    def process_image(self, ts: datetime, frame_idx: int, frame: np.ndarray) -> np.ndarray:
        det_list = self.detector.detect(frame, frame_index=frame_idx)
        track_events = self.tracker.track(frame, frame_idx, det_list)

        # #kwlee
        # import cv2
        # canvas = frame.copy()
        # for track in list(tracker.tracks)[:5]:
        #     loc = track.location
        #     canvas = loc.draw(canvas, color=color.BLUE, line_thickness=1)
        #     canvas = plot_utils.draw_label(canvas, str(track.track_id), loc.tl.astype(int),
        #                                     color.BLACK, color.BLUE, 2)
        # cv2.imshow("output", canvas)
        # cv2.waitKey(5)
        # for idx, det in enumerate(det_list):
        #     canvas = det.draw(canvas, color=color.YELLOW, line_thickness=1)
        #     canvas = plot_utils.draw_label(canvas, str(idx), det.br.astype(int),
        #                                     color.BLACK, color.YELLOW, 2)
        # cv2.imshow("output", canvas)
        # cv2.waitKey(5)
        
        if self.callback:
            self.callback.tracked(frame, frame_idx, det_list, track_events)

        if self.window_name:
            for det in det_list:
                frame = det.draw(frame, color.WHITE, line_thickness=2)
            for track in track_events:
                if track.is_confirmed():
                    frame = track.draw(frame, color.BLUE, trail_color=color.RED, label_color=color.WHITE)
                elif track.is_temporarily_lost():
                    frame = track.draw(frame, color.BLUE, trail_color=color.LIGHT_GREY, label_color=color.WHITE)
                elif track.is_tentative():
                    frame = track.draw(frame, color.RED)

        return frame

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.show_label = not self.show_label
        
        return key

class TrackWriter(TrackerCallback):
    def __init__(self, track_file: Path) -> None:
        super().__init__()
        self.track_file = track_file
        self.out_handle = None

    def track_started(self) -> None:
        super().track_started()
        self.out_handle = open(self.track_file, 'w')

    def track_stopped(self) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None
        super().track_stopped()

    def tracked(self, frame, frame_idx: int, detections: List[Detection], tracks: List[Track]) -> None:
        for track in tracks:
            self.out_handle.write(self._to_string(frame_idx, track) + '\n')

    def _to_string(self, frame_idx: int, track: Track) -> str:
        tlbr = track.location.tlbr
        return (f"{frame_idx},{track.id},{tlbr[0]:.3f},{tlbr[1]:.3f},{tlbr[2]:.3f},{tlbr[3]:.3f},"
                f"{track.state.value},-1,-1,-1")

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--output", help="detection output file.", required=False)
    parser.add_argument("--show", help="show detections.", action="store_true")
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    return parser.parse_args()


if __name__ == '__main__':
    args = parse_args()

    # open target video file temporarily to find fps, which will be used
    # in calculating 'max_age'
    capture = VideoFileCapture(Path(args.input))
    with capture:
        max_age = int(capture.fps) * 3

    detector = DetectorLoader.load(args.detector)

    dna_home_dir = Path(args.home)
    model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model640.pt'
    # model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model280.pt'
    # model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'best.pt'
    tracker = DeepSORTTracker(weights_file=model_file.absolute(),
                                matching_threshold=args.match_score,
                                max_iou_distance=args.max_iou_distance,
                                max_age=max_age)
    track_writer = TrackWriter(args.output) if args.output else None

    display_window_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, detector, tracker, track_writer, window_name=display_window_name,
                                show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )