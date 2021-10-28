from typing import List, Union
import numpy as np
import cv2

def localize_bbox(pt, K=np.eye(3), distort=None, cam_ori=np.eye(3), cam_pos=np.zeros((3, 1)), offset=0.):
    if len(pt) == 4: # [tl.x, tl.y, br.x, br.y]
        tl_x, tl_y, br_x, br_y = pt
        foot_p = [(tl_x + br_x) / 2, br_y]
        head_p = [(tl_x + br_x) / 2, tl_y]

        foot_n, head_n = cv2.undistortPoints(np.array([foot_p, head_p]), K, distort).squeeze(axis=1)
        foot_c = np.matmul(cam_ori, np.append(foot_n, 1))
        head_c = np.matmul(cam_ori, np.append(head_n, 1))

        scale = (offset - cam_pos[1]) / foot_c[1]
        position = scale * foot_c + cam_pos
        height   = scale * (foot_c[1] - head_c[1])
        distance = scale * np.linalg.norm(foot_c)
        return (position, height, distance)
    return None


K = np.array([[1096.54,       0, 967.197],
                [      0, 1098.89, 553.595],
                [      0,       0,       1]])
DISTORT = np.array([-0.328953, 0.0743138, 0.00229938, 0.000639402])
ORI = np.array([[-0.203131, 0.232303, -0.951196],
                [-0.10444,  0.960766,  0.256944],
                [ 0.973566, 0.151536, -0.170899]])
POS = np.array([57.3509, -4.99182, 28.083])


from omegaconf import OmegaConf

from dna import Box, utils as dna_utils
from dna.camera import ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.det import DetectorLoader
from dna.track import LogFileBasedObjectTracker, ObjectTracker, ObjectTrackingProcessor, TrackerCallback, Track
from dna.platform import DNAPlatform

class HandleGlobalCoords(TrackerCallback):
    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass
    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        print(f"frame_idx={frame_idx}")
        for track in tracks:
            res = localize_bbox(track.location.tlbr, K, DISTORT, ORI, POS)
            print(res[0])

if __name__ == '__main__':
    conf = OmegaConf.load("conf/config.yaml")

    uri = "etri:05"
    cap_type = image_capture_type(uri)
    if cap_type == ImageCaptureType.PLATFORM:
        platform = DNAPlatform.load_from_config(conf.platform)
        _, camera_info = platform.get_resource("camera_infos", (uri,))
        uri = camera_info.uri
        blind_regions = camera_info.blind_regions
    else:
        blind_regions = None
    cap = load_image_capture(uri)

    tracker = LogFileBasedObjectTracker("C:/Temp/data/etri/etri_05_track.txt")
    handler = HandleGlobalCoords()

    display_window_name = "output"
    with ObjectTrackingProcessor(cap, tracker, handler,
                                window_name=display_window_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed