from __future__ import annotations
from typing import List
from pathlib import Path

import numpy as np

from dna import color
from dna.camera import ImageProcessor
from .detector import ObjectDetector
from .types import Detection


class ObjectDetectingProcessor(ImageProcessor):
    def __init__(self, capture, detector: ObjectDetector,
                    window_name: str=None, output_video: Path=None, output: Path=None,
                    show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, output_video=output_video,
                            show_progress=show_progress)
        self.detector = detector
        self.box_color = color.RED
        self.label_color = color.WHITE
        self.show_score = True
        self.output = output
        self.out_handle = None

    def on_started(self, capture) -> None:
        if self.output:
            self.out_handle = open(self.output, 'w')
        return self

    def on_stopped(self) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        for det in self.detector.detect(frame, frame_idx):
            if self.out_handle:
                self.out_handle.write(self._to_string(frame_idx, det) + '\n')
            if self.window_name or self.output_video:
                frame = det.draw(frame, color=self.box_color, label_color=self.label_color,
                                    show_score=self.show_score)

        return frame

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.label_color = None if self.label_color else color.WHITE
        elif key == ord('c'):
            self.show_score = not self.show_score
        
        return key

    def _to_string(self, frame_idx: int, det: Detection) -> str:
        tlbr = det.tlbr
        return (f"{frame_idx},-1,{tlbr[0]:.3f},{tlbr[1]:.3f},{tlbr[2]:.3f},{tlbr[3]:.3f},"
                f"{det.score:.3f},-1,-1,-1,{det.label}")