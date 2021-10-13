
from datetime import datetime
from pathlib import Path

import cv2

import numpy as np

from dna import VideoFileCapture, ImageProcessor

cap = VideoFileCapture(Path("C:/Temp/data/channel05_9.mp4"))
cap.open()

fourcc = cv2.VideoWriter_fourcc(*'MP4V')
size = (cap.size.width, cap.size.height)
out = cv2.VideoWriter('C:/Temp/channel05_9.mp4', fourcc, 10.0, size)

while cap.is_open():
    _, idx, mat = cap.capture()
    if mat is None:
        break

    if (idx % 6) == 1:
        out.write(mat)
        # cv2.imshow("output", mat)
        # c = cv2.waitKey(1) & 0xFF
        # if c == ord('q'):
        #     break
out.release()
cap.close()