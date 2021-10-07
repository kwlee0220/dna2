from typing import List
from abc import ABCMeta, abstractmethod
from datetime import datetime
import numpy as np

from dna import ImageProcessor, ImageCapture, color
from . import ObjectTracker, DetectionBasedObjectTracker
from .track_callbacks import TrackerCallback, TrailCollector, DemuxTrackerCallback
from .utils import draw_track_trail


class ObjectTrackingProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, tracker: ObjectTracker, callback: TrackerCallback,
                window_name:str=None, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, show_progress=show_progress)

        self.tracker = tracker
        self.is_detection_based = isinstance(self.tracker, DetectionBasedObjectTracker)
        self.trail_collector = TrailCollector()
        self.callback = DemuxTrackerCallback([self.trail_collector, callback])  \
                            if callback else self.trail_collector
        self.show_label = True

    def on_started(self) -> None:
        if self.callback:
            self.callback.track_started(self.tracker)
        return self

    def on_stopped(self) -> None:
        if self.callback:
            self.callback.track_stopped(self.tracker)

    def process_image(self, ts: datetime, frame_idx: int, frame: np.ndarray) -> np.ndarray:
        track_events = self.tracker.track(frame, frame_idx)
        if self.callback:
            self.callback.tracked(self.tracker, frame, frame_idx, track_events)

        if self.window_name:
            if self.is_detection_based:
                for det in self.tracker.last_frame_detections():
                    frame = det.draw(frame, color.WHITE, line_thickness=2)

            for track in track_events:
                trail = [mark.location for mark in self.trail_collector.get_trail(track.id, [])]
                if track.is_confirmed():
                    frame = draw_track_trail(frame, track, color.BLUE, label_color=color.WHITE,
                                            trail=trail, trail_color=color.RED)
                if track.is_temporarily_lost():
                    frame = draw_track_trail(frame, track, color.BLUE, label_color=color.WHITE,
                                            trail=trail, trail_color=color.LIGHT_GREY)
                elif track.is_tentative():
                    frame = draw_track_trail(frame, track, color.RED, label_color=color.WHITE,
                                            trail=trail, trail_color=color.BLUE)

        return frame

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.show_label = not self.show_label
        
        return key