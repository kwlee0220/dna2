
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
        pt = self.path[self.index][0]
        frame = cv2.circle(frame, pt.xy.astype(int), 7, color.RED, thickness=-1, lineType=cv2.LINE_AA)
        frame = plot_utils.draw_label(frame, str(self.traj.luid), pt.xy.astype(int), color.RED, color.WHITE, 2)
        self.index += 1

        return frame

    def set_control(self, key: int) -> int:
        if key == ord('l'):
            self.show_label = not self.show_label
        
        return key

platform = DNAPlatform()
platform.connect()

# camera_info_rset = platform.get_resource_set('camera_infos')
# print(camera_info_rset.get_all())

# info = CameraInfo("other:1", 1, Size2i([1024, 768]))
# camera_info_rset.insert(info)
# print(camera_info_rset.get_all())

trajectories = platform.get_resource_set('trajectories')
# for traj in trajectories.get_where("luid=84 and path_count > 50"):
#     print(traj)
traj = trajectories.get_all(cond_expr="luid=84 and path_count > 50")[0]
print(traj)

camera_info_rset = platform.get_resource_set('camera_infos')
camera_info = camera_info_rset.get((traj.camera_id,))
print(camera_info)

cap = VideoFileCapture(Path('C:/Temp/data/cam_1.mp4'),
                            begin_frame=traj.first_frame, end_frame=traj.last_frame)
with TrajectoryDisplayProcessor(cap, traj) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )