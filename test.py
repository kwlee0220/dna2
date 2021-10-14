
from datetime import datetime
from pathlib import Path

import cv2

import math
import numpy as np

from dna import Point, Size2i, Size2d
from dna.platform import CameraInfo, DNAPlatform, LocalPath
from dna.camera import DefaultImageCapture, ImageProcessor, VideoFileCapture, SyncImageCapture

# platform = DNAPlatform()
# platform.connect()

# camera_info_rset = platform.get_resource_set('camera_infos')
# camera_info = camera_info_rset.get(("ai_city:9",))
# max_diff = camera_info.size / 500
# print(camera_info, max_diff)

# local_paths = platform.get_resource_set('local_paths')
# path: LocalPath = local_paths.get(('ai_city:9', 228))
# print(path)

# cap = DefaultImageCapture("rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/3/media.smp",
#                     target_size=Size2i(800,600))
cap = VideoFileCapture(Path("C:/Temp/data/cam_9.mp4"), begin_frame=1000, end_frame=1500)
cap = SyncImageCapture(cap)
# with ImageProcessor(cap, window_name="output") as proc:
#     proc.run()
cap.open()
while cap.is_open():
    _, frame_idx, frame = cap.capture()

    cv2.imshow("output", frame)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('q'):
        break
cap.close()