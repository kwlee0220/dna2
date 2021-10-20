from pathlib import Path
import time
from datetime import datetime
import logging

import numpy as np
from omegaconf import OmegaConf

import dna
from dna import color
from dna.camera import ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.det import DetectorLoader, ObjectDetector, Detection
from dna.platform import DNAPlatform


class ObjectDetectingProcessor(ImageProcessor):
    def __init__(self, capture, detector: ObjectDetector,
                    window_name: str=None, output_video: Path=None,
                    output: Path=None, show_progress=False) -> None:
        super().__init__(capture, window_name=window_name, output_video=output_video,
                            show_progress=show_progress)
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

    def process_image(self, frame: np.ndarray, frame_idx: int, ts: datetime) -> np.ndarray:
        for det in self.detector.detect(frame, frame_idx):
            if self.out_handle:
                self.out_handle.write(self._to_string(frame_idx, det) + '\n')
            if self.window_name or self.output_video:
                frame = det.draw(frame, color=color.RED, label_color=self.label_color,
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


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect objects in an video")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=dna.DNA_CONIFIG_FILE)
    parser.add_argument("--sync", help="sync to fps", action="store_true")
    parser.add_argument("--begin_frame", type=int, metavar="<number>", help="the first frame index (from 1)", default=1)
    parser.add_argument("--end_frame", type=int, metavar="<number>", help="the last frame index", default=None)

    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--output", metavar="file",
                        help="output detection file.", required=False)
    parser.add_argument("--output_video", metavar="file",
                        help="output video file", required=False)
    parser.add_argument("--show_progress", help="show progress bar.", action="store_true")
    parser.add_argument("--show", help="show detections.", action="store_true")
    return parser.parse_known_args()


if __name__ == '__main__':
    args, unknown = parse_args()

    uri = args.camera
    cap_type = image_capture_type(uri)
    if cap_type == ImageCaptureType.PLATFORM:
        conf = OmegaConf.load(args.conf)
        platform = DNAPlatform.load_from_config(conf.platform)
        _, camera_info = platform.get_resource("camera_infos", (uri,))
        uri = camera_info.uri
    cap = load_image_capture(uri, sync=args.sync, begin_frame=args.begin_frame, end_frame=args.end_frame)
    
    detector = DetectorLoader.load(args.detector)
    window_name = "output" if args.show else None
    with ObjectDetectingProcessor(cap, detector, window_name=window_name, output_video=args.output_video,
                                    output=args.output, show_progress=args.show_progress) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta
        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}/s" )