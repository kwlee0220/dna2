from typing import List, Union
from dataclasses import dataclass
from collections import defaultdict

import numpy as np

from dna import color, Box, plot_utils
from dna.camera import ImageProcessor, ImageCapture
from .types import TrackState, Track
from .tracker import ObjectTracker, DetectionBasedObjectTracker
from .track_callbacks import TrackerCallback, DemuxTrackerCallback


class TrailCollector(TrackerCallback):
    @dataclass(frozen=True, unsafe_hash=True)
    class Mark:
        state: TrackState
        location: Box

        def __repr__(self) -> str:
            return f"[{self.location}]-"

    def __init__(self) -> None:
        super().__init__()
        self.tracks = defaultdict(list)

    def get_trail(self, track_id: str, def_value=None) -> Union[List[Mark], None]:
        return self.tracks.get(track_id, def_value)

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass

    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            if track.state == TrackState.Confirmed  \
                or track.state == TrackState.TemporarilyLost    \
                or track.state == TrackState.Tentative:
                mark = TrailCollector.Mark(state=track.state, location=track.location)
                self.tracks[track.id].append(mark)
            elif track.state == TrackState.Deleted:
                self.tracks.pop(track.id, None)


def draw_track_trail(mat, track: Track, color, label_color=None,
                    trail: List[Box]=None, trail_color=None, line_thickness=2) -> np.ndarray:
    mat = track.draw(mat, color, label_color=label_color, line_thickness=2)
    if trail_color:
        mat = plot_utils.draw_line_string(mat, [bbox.center() for bbox in trail[-11:]],
                                        trail_color, line_thickness)
    return mat


class ObjectTrackingProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, tracker: ObjectTracker, callback: TrackerCallback=None,
                window_name:str=None, output_video=None, show_zones=False, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, output_video=output_video,
                        show_progress=show_progress)

        self.tracker = tracker
        self.is_detection_based = isinstance(self.tracker, DetectionBasedObjectTracker)
        self.trail_collector = TrailCollector()
        self.callback = DemuxTrackerCallback([self.trail_collector, callback])  \
                            if callback else self.trail_collector
        self.show_zones = show_zones

    def on_started(self, capture) -> None:
        if self.callback:
            self.callback.track_started(self.tracker)

    def on_stopped(self) -> None:
        if self.callback:
            self.callback.track_stopped(self.tracker)

    def set_control(self, key: int) -> int:
        if key == ord('r'):
            self.show_zones = not self.show_zones
        return key

    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        tracks = self.tracker.track(frame, frame_idx, ts)
        if self.callback:
            self.callback.tracked(self.tracker, frame, frame_idx, tracks)

        if self.window_name or self.output_video:
            if self.show_zones:
                for region in self.tracker.params.blind_zones:
                    frame = region.draw(frame, color.MAGENTA, 2)
                for region in self.tracker.params.dim_zones:
                    frame = region.draw(frame, color.RED, 2)

            if self.is_detection_based:
                for det in self.tracker.last_frame_detections():
                    frame = det.draw(frame, color.WHITE, line_thickness=2)

            for track in tracks:
                if track.is_tentative():
                    trail = [mark.location for mark in self.trail_collector.get_trail(track.id, [])]
                    frame = draw_track_trail(frame, track, color.RED, label_color=color.WHITE,
                                            trail=trail, trail_color=color.BLUE)
            for track in sorted(tracks, key=lambda t: t.id, reverse=True):
                if not track.is_tentative():
                    trail = [mark.location for mark in self.trail_collector.get_trail(track.id, [])]
                    if track.is_confirmed():
                        frame = draw_track_trail(frame, track, color.BLUE, label_color=color.WHITE,
                                                trail=trail, trail_color=color.RED)
                    if track.is_temporarily_lost():
                        frame = draw_track_trail(frame, track, color.BLUE, label_color=color.WHITE,
                                                trail=trail, trail_color=color.LIGHT_GREY)

        return frame