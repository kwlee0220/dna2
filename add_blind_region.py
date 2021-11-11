from __future__ import annotations

from omegaconf import OmegaConf
import cv2
import numpy as np
import threading
import time

from dna import color, Point, Box, Size2d, DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.camera import Camera, ImageCapture
from dna.platform import DNAPlatform


class BoxSelector:
    def __init__(self, bg_image) -> None:
        self.tl = None
        self.box = None
        self.bg_image = bg_image
        self.done = False

        self.convas = bg_image.copy()
        cv2.imshow("image", self.convas)
        cv2.waitKey(1)

    @classmethod
    def zeros(cls, size: Size2d) -> BoxSelector:
        return BoxSelector(np.zeros((size.height, size.width, 3), np.uint8))

    @classmethod
    def from_camera(cls, camera: ImageCapture) -> BoxSelector:
        camera.open()
        try:
            _, _, frame = camera.capture()
            return BoxSelector(frame)
        finally:
            camera.close()

    def mouse_callback(self, event, x, y, flags, param):
        pt = Point(x, y)

        if self.tl:
            self.convas = self.bg_image.copy()

        if event == cv2.EVENT_LBUTTONDOWN:
            self.tl = pt
            self.box = None
        elif event == cv2.EVENT_LBUTTONUP:
            tl_x, br_x = (self.tl.x, pt.x) if self.tl.x <= pt.x else (pt.x, self.tl.x)
            tl_y, br_y = (self.tl.y, pt.y) if self.tl.y <= pt.y else (pt.y, self.tl.y)
            
            self.box = Box.from_tlbr(np.array([tl_x, tl_y, br_x, br_y]))
            self.tl = None

        if self.box:
            self.box.draw(self.convas, color.RED, line_thickness=1)
        elif self.tl:
            box = Box.from_points(self.tl, pt)
            box.draw(self.convas, color.RED, line_thickness=1)

        cv2.imshow("image", self.convas)

    def run(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", selector.mouse_callback)
        while not self.done:
            key = cv2.waitKey(1) & 0xFF
            if key == 13:
                self.done = True
            elif key == 27:
                self.box = None
                self.done = True

        cv2.destroyWindow("image")

        return self.box


import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Generating tracking events from a track-file")
    parser.add_argument("node", metavar='node_id', help="target node_id")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)
    parser.add_argument("--begin_frame", type=int, metavar="<number>",
                        help="the first frame index (from 1)", default=5)
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE, args.node)

    platform = DNAPlatform.load_from_config(conf.platform)

    camera_info = Camera.from_conf(conf.camera)
    cap = camera_info.get_capture(sync=False, begin_frame=args.begin_frame)
    cap.open()
    _,_,bg_img = cap.capture()
    cap.close()


    blind_zones, dim_zones = [], []
    if conf.tracker.get("blind_zones", None):
        blind_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in conf.tracker.blind_zones]
    if conf.tracker.get("dim_zones", None):
        dim_zones = [Box.from_tlbr(np.array(zone, dtype=np.int32)) for zone in conf.tracker.dim_zones]

    for box in blind_zones:
        bg_img = box.draw(bg_img, color.GREEN)
    for box in dim_zones:
        bg_img = box.draw(bg_img, color.BLUE)
    selector = BoxSelector(bg_img)
    box = selector.run()
    if box and box.is_valid():
        print("new region=", list(box.tlbr))