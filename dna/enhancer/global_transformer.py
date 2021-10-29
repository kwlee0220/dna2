from typing import List, Union
import numpy as np
import cv2


from omegaconf import OmegaConf

from dna import Box, utils as dna_utils
from dna.camera import ImageProcessor, ImageCaptureType, image_capture_type, load_image_capture
from dna.det import DetectorLoader
from dna.track import LogFileBasedObjectTracker, ObjectTracker, ObjectTrackingProcessor, TrackerCallback, Track
from dna.platform import DNAPlatform

CONVAS = None

class HandleGlobalCoords(TrackerCallback):
    def __init__(self, camera_geometry, toptopview_view) -> None:
        super().__init__()
        self.geometry = camera_geometry
        self.topview = topview

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None: pass
    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        print(f"frame_idx={frame_idx}")
        for track in tracks:
            res = localize_bbox(track.location.tlbr, self.geometry['K'], self.geometry['distort'],
                                self.geometry['ori'], self.geometry['pos'])
            # print(res[0])

            px = np.array(conv_meter2pixel(res[0], self.topview['origin'],
                                            self.topview['meter_per_pixel'])).astype(int)
            print(px)

import pickle, sys
if __name__ == '__main__':
    conf = OmegaConf.load("conf/config.yaml")

    with open('camera_etri_test.pickle', 'rb') as f:
        topview, cameras = pickle.load(f)
    if not topview or not cameras:
        sys.exit('Error: The camera file contains no camera information.')

    topview['convas'] = cv2.imread(topview['file'])

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
    handler = HandleGlobalCoords(cameras[1], topview)

    display_window_name = "output"
    with ObjectTrackingProcessor(cap, tracker, handler,
                                window_name=display_window_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed