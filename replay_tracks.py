from typing import List
from pathlib import Path
from datetime import datetime
import cv2
import numpy as np

from dna import ImageProcessor, VideoFileCapture
from dna import color
from dna.track import ObjectTracker, LogFileBasedObjectTracker, TrackerCallback


class ReplayingTrackProcessor(ImageProcessor):
    def __init__(self, capture, tracker: ObjectTracker, callback: TrackerCallback,
                window_name:str=None, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, show_progress=show_progress)

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
        track_events = self.tracker.track(frame, frame_idx, [])
        
        if self.callback:
            self.callback.tracked(frame, frame_idx, [], track_events)

        if self.window_name:
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


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generating tracking events from a track-file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--track_file", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--video_file", help="input source.", required=True)
    parser.add_argument("--show", help="show detections.", action="store_true")
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    # open target video file temporarily to find fps, which will be used
    # in calculating 'max_age'
    capture = VideoFileCapture(Path(args.video_file))

    dna_home_dir = Path(args.home)
    tracker = LogFileBasedObjectTracker(args.track_file)

    display_window_name = "output" if args.show else None
    with ReplayingTrackProcessor(capture, tracker, callback=None,window_name=display_window_name,
                                show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )