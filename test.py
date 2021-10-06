from pathlib import Path
import cv2

from dna import plot_utils
from dna.det import DetectorLoader

detector = DetectorLoader.YOLOv4Torch(Path('./dna/det/yolov4_torch').absolute())
# detector = DetectorFactory.YOLOv5Torch(Path('./dna/det/yolov5/weights/yolov5s.pt').absolute())

image_dir = Path('./images')
print(f'loading images from {image_dir.absolute()}')
for image in image_dir.glob('*.jpg'):
    mat = cv2.imread(str(image))
    det_list = detector.detect(mat)

    # print(det_list)
    for det in det_list:
        plot_utils.plot_detection(mat, det)
    cv2.imshow('output', mat)
    key = cv2.waitKey(5000)
    if key == ord('q'):
        break
cv2.destroyAllWindows()