
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
import cv2

import dna
from dna import color, plot_utils
from dna.camera import ImageCapture, ImageProcessor, load_image_capture
from dna.platform import DNAPlatform, LocalPath

class TrackShow:
    def __init__(self, path: LocalPath) -> None:
        self.path = path
        self.points = path.points
        self.index = 0

    def show(self, convas, frame_idx: int):
        idx = frame_idx - self.path.first_frame
        if idx >= 0 and idx < len(self.points):
            start_idx = max(idx - 10, 0)
            plot_utils.draw_line_string(convas, self.points[start_idx:idx], color.RED)

            pt = self.points[idx]
            convas = cv2.circle(convas, pt.xy.astype(int), 5, color.RED, thickness=-1, lineType=cv2.LINE_AA)
            convas = plot_utils.draw_label(convas, str(self.path.luid), pt.xy.astype(int), color.BLACK, color.RED, 3)
        return convas

class TrackShowProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, paths: List[LocalPath], output_video: Path=None) -> None:
        super().__init__(capture, window_name='output', output_video=output_video,
                            show_progress=False, stop_at_the_last=True)
        self.shows = [TrackShow(path) for path in paths]
        self.shows.sort(key=TrackShowProcessor.get_first_frame)
        self.start_idx = 0

    def process_image(self, convas, frame_idx: int, ts):
        sidx = self.start_idx
        for idx in range(self.start_idx, len(self.shows)):
            show = self.shows[idx]
            if frame_idx > show.path.last_frame:
                sidx = idx + 1
            else:
                break
        
        self.start_idx = sidx
        for show in self.shows[self.start_idx:]:
            if frame_idx < show.path.first_frame:
                break
            show.show(convas, frame_idx)

        return convas

    @staticmethod
    def get_first_frame(show: TrackShow):
        return show.path.first_frame


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a localpath on the screen")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=dna.DNA_CONIFIG_FILE)
    parser.add_argument("--no_sync", help="do not sync to fps", action="store_true")

    parser.add_argument("--output_video", metavar="file", help="output video file", required=False)
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()

    conf = OmegaConf.load(args.conf)
    config_grp = dna.parse_config_args(unknown)
    platform = DNAPlatform.load_from_config(conf.platform)

    rset = platform.get_resource_set("local_paths")
    lpaths = rset.get_all(f"camera_id='{args.camera}'")

    _, camera_info = platform.get_resource("camera_infos", (args.camera,))

    cap = load_image_capture(camera_info.uri, sync=not args.no_sync)
    with TrackShowProcessor(cap, lpaths, output_video=args.output_video) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )