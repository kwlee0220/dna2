
from collections import namedtuple
from datetime import datetime
from pathlib import Path

import cv2
import pickle

import math
import numpy as np

from dna.enhancer.world_transform import CameraGeometry

def _to_geometry(camera) -> CameraGeometry:
    return CameraGeometry(camera['K'], camera['distort'], camera['ori'], camera['pos'])

with open('camera_etri_test.pickle', 'rb') as f:
    topview, cameras = pickle.load(f)

with open('etri_04.pickle', 'wb') as f:
    pickle.dump(_to_geometry(cameras[0]), f)

with open('etri_05.pickle', 'wb') as f:
    pickle.dump(_to_geometry(cameras[1]), f)

with open('etri_05.pickle', 'rb') as f:
    o = pickle.load(f)

with open('etri_06.pickle', 'wb') as f:
    pickle.dump(_to_geometry(cameras[2]), f)

with open('etri_07.pickle', 'wb') as f:
    pickle.dump(_to_geometry(cameras[3]), f)