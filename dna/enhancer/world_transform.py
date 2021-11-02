from typing import List, Union
from collections import namedtuple

import pickle
from queue import Queue
from omegaconf import OmegaConf
import cv2
import numpy as np
from shapely.geometry import Point as ShapelyPoint

from dna.platform import DNAPlatform
from .types import TrackEvent, end_of_track_event
from .event_processor import EventProcessor

CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos')


_CHANNEL = "world_coords"
class WorldTransform(EventProcessor):
    def __init__(self, camera_id, pubsub, in_queue: Queue, conf: OmegaConf) -> None:
        self.camera_id = camera_id
        self.in_queue = in_queue
        self.pubsub = pubsub
        with open(conf.file, 'rb') as f:
            self.geometry = pickle.load(f)

    def close(self) -> None:
        self.pubsub.publish(_CHANNEL, end_of_track_event(self.camera_id))

    def subscribe(self) -> Queue:
        return self.pubsub.subscribe(_CHANNEL)

    def handle_event(self, ev: TrackEvent) -> None:
        wcoord, height, dist = self._localize_bbox(ev.location.tlbr)

        enhanced = TrackEvent(ev.camera_id, ev.luid, ev.location,
                                world_coord=ShapelyPoint(wcoord), distance=dist,
                                frame_index=ev.frame_index, ts=ev.ts)
        self.pubsub.publish(_CHANNEL, enhanced)

    def _localize_bbox(self, tlbr, offset=0):
        tl_x, tl_y, br_x, br_y = tlbr
        foot_p = [(tl_x + br_x) / 2, br_y]
        head_p = [(tl_x + br_x) / 2, tl_y]

        foot_n, head_n = cv2.undistortPoints(np.array([foot_p, head_p]), self.geometry.K,
                                            self.geometry.distort).squeeze(axis=1)
        foot_c = np.matmul(self.geometry.ori, np.append(foot_n, 1))
        head_c = np.matmul(self.geometry.ori, np.append(head_n, 1))

        scale = (offset - self.geometry.pos[1]) / foot_c[1]
        position = scale * foot_c + self.geometry.pos
        height   = scale * (foot_c[1] - head_c[1])
        distance = scale * np.linalg.norm(foot_c)
        return (position, height, distance)