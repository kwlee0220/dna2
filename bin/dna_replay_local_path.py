
from pathlib import Path

from omegaconf import OmegaConf
import cv2

import dna
from dna import color, plot_utils
from dna.camera import ImageCapture, ImageProcessor, load_image_capture
from dna.platform import DNAPlatform, LocalPath


class LocalPathDisplayProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, path: LocalPath, output_video: Path=None) -> None:
        super().__init__(capture, window_name='output', output_video=output_video,
                            show_progress=False, stop_at_the_last=True)

        self.path = path
        self.points = path.points
        self.index = 0

    def process_image(self, convas, frame_idx: int, ts):
        convas = plot_utils.draw_line_string(convas, self.points[self.index:], color.GREEN)
        if frame_idx >= self.path.first_frame and frame_idx <= self.path.last_frame:
            convas = plot_utils.draw_line_string(convas, self.points[0:self.index+1], color.RED, 3)
            
            pt = self.points[self.index]
            convas = cv2.circle(convas, pt.xy.astype(int), 7, color.RED, thickness=-1, lineType=cv2.LINE_AA)
            convas = plot_utils.draw_label(convas, str(self.path.luid), pt.xy.astype(int), color.BLACK, color.RED, 4)

            self.index += 1

        return convas


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a localpath on the screen")
    parser.add_argument("camera", metavar='camera_uri', help="target camera uri")
    parser.add_argument("--conf", help="DNA framework configuration", default=dna.DNA_CONIFIG_FILE)
    parser.add_argument("--no_sync", help="do not sync to fps", action="store_true")
    parser.add_argument("--id", type=int, help="target object id")

    parser.add_argument("--output_video", metavar="file", help="output video file", required=False)
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()

    conf = OmegaConf.load(args.conf)
    config_grp = dna.parse_config_args(unknown)
    platform = DNAPlatform.load_from_config(conf.platform)

    _, lpath = platform.get_resource("local_paths", (args.camera, args.id))
    if lpath is None:
        print(f"invalid track object: camera_id='{args.camera}', id='{args.id}'", file=sys.stderr)
        exit(-1)
    print(f"path: {lpath.first_frame} -> {lpath.last_frame}")

    _, camera_info = platform.get_resource("camera_infos", (args.camera,))
    margin = int(camera_info.fps / 2)
    begin_frame = max(lpath.first_frame - margin, 1)
    end_frame = lpath.last_frame

    cap = load_image_capture(camera_info.uri, sync=not args.no_sync,
                            begin_frame=begin_frame, end_frame=end_frame)
    with LocalPathDisplayProcessor(cap, lpath,
                            output_video=args.output_video) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )