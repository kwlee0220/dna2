
from pathlib import Path
from typing import List

from omegaconf import OmegaConf
import cv2

from dna import color, plot_utils, DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.camera import ImageCapture, ImageProcessor, Camera
from dna.platform import DNAPlatform, LocalPath

class LocalPathDraw:
    def __init__(self, path: LocalPath) -> None:
        self.path = path
        self.points = path.points
        self.index = 0

    def draw_path(self, convas, frame_idx: int, ts):
        convas = plot_utils.draw_line_string(convas, self.points[self.index:], color.GREEN)
        if frame_idx >= self.path.first_frame and frame_idx <= self.path.last_frame:
            convas = plot_utils.draw_line_string(convas, self.points[0:self.index+1], color.RED, 3)
            
            pt = self.points[self.index]
            convas = cv2.circle(convas, pt.xy.astype(int), 7, color.RED, thickness=-1, lineType=cv2.LINE_AA)
            convas = plot_utils.draw_label(convas, str(self.path.luid), pt.xy.astype(int), color.BLACK, color.RED, 4)

            self.index += 1
        elif frame_idx > self.path.last_frame:
            last_pt = self.points[-1]
            convas = cv2.circle(convas, last_pt.xy.astype(int), 5, color.RED,
                                thickness=-1, lineType=cv2.LINE_AA)
            convas = plot_utils.draw_label(convas, str(self.path.luid), last_pt.xy.astype(int),
                                            color.BLACK, color.RED, 3)

        return convas

class LocalPathDisplayProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, paths: List[LocalPath], output_video: Path=None) -> None:
        super().__init__(capture, window_name='output', output_video=output_video,
                            show_progress=False, stop_at_the_last=True)

        self.draws = [LocalPathDraw(path) for path in paths]

    def process_image(self, convas, frame_idx: int, ts):
        for draw in self.draws:
            convas = draw.draw_path(convas, frame_idx, ts)
        return convas


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a localpath on the screen")
    parser.add_argument("node", metavar='node_id', help="target node_id")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--no_sync", help="do not sync to fps", action="store_true")
    parser.add_argument("--id", type=int, nargs='+', help="target object id")

    parser.add_argument("--output_video", metavar="file", help="output video file", required=False)
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()

    conf = load_config(DNA_CONIFIG_FILE, args.node)
    config_grp = parse_config_args(unknown)

    camera_info = Camera.from_conf(conf.camera)

    platform = DNAPlatform.load_from_config(conf.platform)
    rset = platform.get_resource_set("local_paths")

    lpaths = []
    first_frame = sys.maxsize * 2 + 1
    last_frame = 0
    for id in args.id:
        lpath = rset.get((camera_info.camera_id, id))
        if lpath:
            first_frame = min(first_frame, lpath.first_frame)
            last_frame = max(last_frame, lpath.last_frame)
        else:
            print(f"invalid track object: camera_id='{camera_info.camera_id}', id='{args.id}'", file=sys.stderr)
            exit(-1)
        lpaths.append(lpath)
    print(f"path: {first_frame} -> {last_frame}")

    begin_frame = max(first_frame - 5, 1)
    end_frame = last_frame
    cap = camera_info.get_capture(sync=not args.no_sync, begin_frame=begin_frame, end_frame=end_frame)
    with LocalPathDisplayProcessor(cap, lpaths,
                            output_video=args.output_video) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )