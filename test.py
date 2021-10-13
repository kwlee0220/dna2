
from datetime import datetime
from pathlib import Path

import cv2

import numpy as np

from dna.platform import CameraInfo, DNAPlatform, Trajectory

platform = DNAPlatform()
platform.connect()

camera_info_rset = platform.get_resource_set('camera_infos')
camera_info = camera_info_rset.get(("ai_city:9",))

trajectories = platform.get_resource_set('trajectories')
# trajs = trajectories.get_all(cond_expr=f"camera_id='ai_city:9' and luid=1")
traj = trajectories.get(('ai_city:9', 1))
print(traj)