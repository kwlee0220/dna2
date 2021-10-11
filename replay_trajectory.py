
from datetime import datetime
from pathlib import Path

import cv2

from dna import ImageProcessor, ImageCapture, color, plot_utils
from dna.image_capture import VideoFileCapture
from dna.platform import CameraInfo, DNAPlatform, Trajectory
from dna.types import Size2i


class TrajectoryDisplayProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, traj: Trajectory) -> None:
        super().__init__(capture, window_name='output', show_progress=False)

        self.traj = traj
        self.path = traj.path
        self.index = 0
        self.show_label = True

    def on_started(self) -> None:
        pass

    def on_stopped(self) -> None:
        pass

    def process_image(self, frame, frame_idx: int, ts: datetime):
        pt = self.path[self.index]
        frame = cv2.circle(frame, pt.xy.astype(int), 7, color.RED, thickness=-1, lineType=cv2.LINE_AA)
        frame = plot_utils.draw_label(frame, str(self.traj.luid), pt.xy.astype(int), color.RED, color.WHITE, 2)
        self.index += 1

        return frame

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.show_label = not self.show_label
        
        return key


import sys
import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Replay a trajectory on the screen")
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

    trajectories = platform.get_resource_set('trajectories')
    trajs = trajectories.get_all(cond_expr=f"camera_id='{args.camera_id}' and luid={args.luid}")
    if len(trajs) != 1:
        print(f"invalid track object: camera_id='{args.camera_id}', luid='{args.luid}'", file=sys.stderr)
        exit(-1)
    traj = trajs[0]

    capture = VideoFileCapture(Path(args.input),
                                begin_frame=traj.first_frame, end_frame=traj.last_frame)
    with TrajectoryDisplayProcessor(capture, traj) as processor:
            from timeit import default_timer as timer
            from datetime import timedelta

            started = timer()
            frame_count = processor.run()
            elapsed = timer() - started
            fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )