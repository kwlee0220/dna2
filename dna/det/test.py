from pathlib import Path
import cv2

import sys
print(sys.path)
from dna import plot_utils

# from dna.det import Yolov5Detector
# detector = Yolov5Detector()
from dna.det import Yolov4TorchDetector
detector = Yolov4TorchDetector()

image_dir = Path('./images')
print(f'loading images from {image_dir.absolute()}')
for image in image_dir.glob('*.jpg'):
    mat = cv2.imread(str(image))
    det_list = detector.detect(mat)

    print(det_list)
    for det in det_list:
        plot_utils.plot_detection(mat, det)
    cv2.imshow('output', mat)
    key = cv2.waitKey(5000)
    if key == ord('q'):
        break
cv2.destroyAllWindows()