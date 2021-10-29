from typing import List, Union
from dataclasses import dataclass
from collections import defaultdict

from omegaconf import OmegaConf
import numpy as np
import cv2

from dna import color, Box, plot_utils
from dna.camera import ImageCapture, ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.track import ObjectTracker, Track, LogFileBasedObjectTracker
from dna.platform import DNAPlatform


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

def conv_pixel2meter(pt, origin, meter_per_pixel):
    x = (pt[0] - origin[0]) * meter_per_pixel
    y = 0
    z = (origin[1] - pt[1]) * meter_per_pixel
    return [x, y, z]

def conv_meter2pixel(pt, origin, meter_per_pixel):
    u = pt[0] / meter_per_pixel + origin[0]
    v = origin[1] - pt[2] / meter_per_pixel
    return [u, v]

class TopViewProcessor(ImageProcessor):
    def __init__(self, capture: ImageCapture, tracker: ObjectTracker, camera_geometry, topview) -> None:
        super().__init__(capture, window_name="view")

        self.tracker = tracker
        self.geometry = camera_geometry
        self.convas = cv2.imread(topview['file'])
        self.topview = topview

    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        convas = self.convas.copy()

        tracks = self.tracker.track(frame, frame_idx, ts)
        for track in tracks:
            res = localize_bbox(track.location.tlbr, self.geometry['K'], self.geometry['distort'],
                                self.geometry['ori'], self.geometry['pos'])
            # print(res[0])

            px = np.array(conv_meter2pixel(res[0], self.topview['origin'],
                                            self.topview['meter_per_pixel'])).astype(int)
            print(px)
            cv2.circle(convas, px, 5, color.RED, -1)
        cv2.imshow("output", convas)
        cv2.waitKey(1)

        return frame

import pickle, sys
if __name__ == '__main__':
    conf = OmegaConf.load("conf/config.yaml")

    with open('camera_etri_test.pickle', 'rb') as f:
        topview, cameras = pickle.load(f)
    if not topview or not cameras:
        sys.exit('Error: The camera file contains no camera information.')

    topview['convas'] = cv2.imread(topview['file'])

    uri = "etri:06"
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
    with TopViewProcessor(cap, tracker, cameras[0], topview) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed