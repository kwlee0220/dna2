
from datetime import datetime
from pathlib import Path

import cv2

from dna import color, plot_utils
from dna.camera import ImageCapture, VideoFileCapture, ImageProcessor
from dna.platform import DNAPlatform, LocalPath
from dna import Size2i


class LocalPathDisplayProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, path: LocalPath) -> None:
        super().__init__(capture, window_name='output', show_progress=False,
                            stop_at_the_last=True)

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
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--camera_id", help="camera id")
    parser.add_argument("--luid", type=int, help="target object id")
    parser.add_argument("--input", help="input source.", required=True)

    parser.add_argument("--db_host", help="host name of DNA data platform", default="localhost")
    parser.add_argument("--db_port", type=int, help="port number of DNA data platform", default=5432)
    parser.add_argument("--db_name", help="database name", default="dna")
    parser.add_argument("--db_user", help="user name", default="postgres")
    parser.add_argument("--db_passwd", help="password", default="dna2021")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    platform = DNAPlatform()
    platform.connect()

    camera_info_rset = platform.get_resource_set('camera_infos')
    camera_info = camera_info_rset.get((args.camera_id,))
    if camera_info is None:
        print(f"unknown camera_id: '{args.camera_id}'", file=sys.stderr)
        exit(-1)

    trajectories = platform.get_resource_set('local_paths')
    traj = trajectories.get((args.camera_id, args.luid))
    if traj is None:
        print(f"invalid track object: camera_id='{args.camera_id}', luid='{args.luid}'", file=sys.stderr)
        exit(-1)

    print(f"path: {traj.first_frame} -> {traj.last_frame}")

    margin = int(camera_info.fps / 2)
    begin_frame = max(traj.first_frame - margin, 1)
    end_frame = traj.last_frame

    capture = VideoFileCapture(Path(args.input),  begin_frame=begin_frame, end_frame=end_frame)
    with LocalPathDisplayProcessor(capture, traj) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )