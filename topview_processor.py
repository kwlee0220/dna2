from typing import List, Union
from dataclasses import dataclass
from collections import defaultdict
from pathlib import Path
from collections import namedtuple

from omegaconf import OmegaConf
import numpy as np
import cv2

from dna import Box, DNA_CONIFIG_FILE, parse_config_args, load_config, color
from dna.camera import ImageCapture, ImageProcessor, Camera
from dna.track import ObjectTracker, Track, LogFileBasedObjectTracker
from dna.platform import DNAPlatform


_METER_PER_PIXEL = 0.12345679012345678
_ORIGIN = [126, 503]
CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos')

def localize_bbox(pt, geom: CameraGeometry, offset=0.):
    tl_x, tl_y, br_x, br_y = pt
    foot_p = [(tl_x + br_x) / 2, br_y]
    head_p = [(tl_x + br_x) / 2, tl_y]

    foot_n, head_n = cv2.undistortPoints(np.array([foot_p, head_p]), geom.K, geom.distort).squeeze(axis=1)
    foot_c = np.matmul(geom.ori, np.append(foot_n, 1))
    head_c = np.matmul(geom.ori, np.append(head_n, 1))

    scale = (offset - geom.pos[1]) / foot_c[1]
    position = scale * foot_c + geom.pos
    height   = scale * (foot_c[1] - head_c[1])
    distance = scale * np.linalg.norm(foot_c)
    return (position, height, distance)

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
    def __init__(self, capture: ImageCapture, tracker: ObjectTracker, camera_geometry, map_image) -> None:
        super().__init__(capture, window_name="view")

        self.tracker = tracker
        self.geometry = camera_geometry
        self.map_image = map_image

    def process_image(self, frame: np.ndarray, frame_idx: int, ts) -> np.ndarray:
        convas = self.map_image.copy()

        tracks = self.tracker.track(frame, frame_idx, ts)
        for track in tracks:
            res = localize_bbox(track.location.tlbr, self.geometry)
            # print(res[0])

            px = np.array(conv_meter2pixel(res[0], _ORIGIN, _METER_PER_PIXEL)).astype(int)
            cv2.circle(convas, px, 5, color.RED, -1)
        cv2.imshow("output", convas)
        cv2.waitKey(1)

        return frame

import pickle, sys, os
if __name__ == '__main__':
    # args, unknown = parse_args()
    # config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE, 'etri_05')
    camera_info = Camera.from_conf(conf.camera)
    cap = camera_info.get_capture(sync=True)
    with open('camera_etri_test.pickle', 'rb') as f:
        topview, cameras = pickle.load(f)
    # if not topview or not cameras:
    #     sys.exit('Error: The camera file contains no camera information.')

    with open('data/camera_geoms/etri_05.pickle', 'rb') as f:
        geom = pickle.load(f)

    map_image = cv2.imread('data/ETRI.png')

    tracker = LogFileBasedObjectTracker("C:/Temp/data/etri/etri_05_track.txt")
    with TopViewProcessor(cap, tracker, geom, map_image) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed