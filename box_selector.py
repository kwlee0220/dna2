
import cv2
import numpy as np

from dna import Point, color
from dna.types import BBox

class BoxSelector:
    def __init__(self) -> None:
        self.tl = None
        self.box = None

        self.convas = np.zeros((480, 640, 3), np.uint8)
        cv2.imshow("image", self.convas)
        cv2.waitKey(1)

    def mouse_callback(self, event, x, y, flags, param):
        pt = Point(x, y)

        if self.tl:
            self.convas = np.zeros((480, 640, 3), np.uint8)

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


selector = BoxSelector()
box = selector.run()
print(box)