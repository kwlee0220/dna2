from __future__ import annotations

from omegaconf import OmegaConf
import cv2
import numpy as np
import threading
import time

from dna import Point, color, BBox, Size2i, plot_utils
from dna.camera import ImageCapture, load_image_capture
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
    def zeros(cls, size: Size2i) -> BoxSelector:
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
            
            self.box = BBox.from_tlbr(np.array([tl_x, tl_y, br_x, br_y]))
            self.tl = None

        if self.box:
            self.box.draw(self.convas, color.RED, line_thickness=1)
        elif self.tl:
            box = BBox.from_points(self.tl, pt)
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
    parser.add_argument("--conf", help="DNA framework configuration", default="conf/config.yaml")
    parser.add_argument("--camera_id", metavar="id", help="target camera id")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    conf = OmegaConf.load(args.conf)
    dict = OmegaConf.to_container(conf.platform)

    platform = DNAPlatform.load(dict)
    rset, camera_info = platform.get_resource("camera_infos", (args.camera_id,))
    cap = load_image_capture(camera_info.uri, camera_info.size, sync=False, begin_frame=5)
    cap.open()
    _,_,bg_img = cap.capture()
    cap.close()

    for box in camera_info.blind_regions:
        bg_img = box.draw(bg_img, color.GREEN)
    selector = BoxSelector(bg_img)
    box = selector.run()
    if box and box.is_valid():
        camera_info.add_blind_region(box)
        rset.update_blind_regions(camera_info)