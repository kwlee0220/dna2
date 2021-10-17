from __future__ import annotations
import cv2
import numpy as np

from dna import Point, color, BBox, Size2i
from dna.camera.image_capture import ImageCapture
from dna.camera.video_file_capture import VideoFileCapture

class BoxSelector:
    def __init__(self, bg_image) -> None:
        self.tl = None
        self.box = None
        self.bg_image = bg_image

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
            self.box = BBox.from_points(self.tl, pt)
            self.tl = None

        if self.box:
            self.box.draw(self.convas, color.RED, line_thickness=1)
        elif self.tl:
            box = BBox.from_points(self.tl, pt)
            box.draw(self.convas, color.RED, line_thickness=1)
        cv2.imshow("image", self.convas)
        cv2.waitKey(1)

    def run(self):
        cv2.namedWindow("image")
        cv2.setMouseCallback("image", selector.mouse_callback)
        while True:
            key = cv2.waitKey(-1) & 0xFF
            if key == 27:
                break
        cv2.destroyWindow("image")

        return self.box

cap = VideoFileCapture("C:/Temp/data/channel06_9.mp4")
selector = BoxSelector.from_camera(cap)
box = selector.run()
print(box.tlbr)