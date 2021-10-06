from pathlib import Path
import time
from datetime import datetime
import logging

import numpy as np

from dna import ImageProcessor, VideoFileCapture, color
from dna.det import DetectorLoader, ObjectDetector, Detection


class ObjectDetectingProcessor(ImageProcessor):
    def __init__(self, capture, detector: ObjectDetector,
                    window_name: str=None, output: Path=None, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, show_progress=show_progress)
        self.detector = detector
        self.label_color = color.WHITE
        self.show_score = True
        self.output = output
        self.out_handle = None

    def on_started(self) -> None:
        if self.output:
            self.out_handle = open(self.output, 'w')
        return self

    def on_stopped(self) -> None:
        if self.out_handle:
            self.out_handle.close()
            self.out_handle = None

    def process_image(self, ts: datetime, frame_idx: int, mat: np.ndarray) -> np.ndarray:
        for det in self.detector.detect(mat, frame_idx):
            if self.out_handle:
                self.out_handle.write(self._to_string(frame_idx, det) + '\n')
            if self.window_name:
                mat = det.draw(mat, color=color.RED, label_color=self.label_color, show_score=self.show_score)

        return mat

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


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in an input video file")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--output", help="detection output file.", required=False)
    parser.add_argument("--fps", help="input source fps.", default=-1, type=int, required=False)
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()
    
    capture = VideoFileCapture(Path(args.input), fps=args.fps)
    detector = DetectorLoader.load(args.detector)

    window_name = "output" if args.show else None
    with ObjectDetectingProcessor(capture, detector, window_name=window_name, output=args.output,
                                    show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}/s" )