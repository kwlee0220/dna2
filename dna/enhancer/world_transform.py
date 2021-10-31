from typing import List, Union
from collections import namedtuple

from queue import Queue
from omegaconf import OmegaConf
import cv2
import numpy as np

from dna.platform import DNAPlatform
from .types import TrackEvent

CameraGeometry = namedtuple('CameraGeometry', 'K,distort,ori,pos')


class WorldTransform:
    def __init__(self, mqueue: Queue, camera_geometry) -> None:
        self.mqueue = mqueue
        self.geometry = camera_geometry

    def handle_event(self, ev: TrackEvent) -> None:
        pos, height, dist = self._localize_bbox(ev.location.tlbr)

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