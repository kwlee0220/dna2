from typing import List

import numpy as np
import cv2

from dna import Box, Point

def draw_line(mat, from_pt: Point, to_pt: Point, color, line_thickness=2) -> np.ndarray:
    return draw_line_raw(mat, from_pt.xy.astype(int), to_pt.xy.astype(int), color, line_thickness)

def draw_line_raw(mat, from_pt, to_pt, color, line_thickness=2) -> np.ndarray:
    return cv2.line(mat, from_pt, to_pt, color, line_thickness)

def draw_line_string_raw(convas, pts: List[List[int]], color, line_thickness=2) -> np.ndarray:
    for pt1, pt2 in zip(pts, pts[1:]):
        convas = draw_line_raw(convas, pt1, pt2, color, line_thickness)
    return convas

def draw_line_string(mat, pts: List[Point], color, line_thickness=2) -> np.ndarray:
    return draw_line_string_raw(mat, [pt.xy.astype(int) for pt in pts], color, line_thickness)

def draw_label(mat, label, tl, color=(225,255,255), fill_color=None, thickness=2) -> np.ndarray:
    txt_thickness = max(thickness - 1, 1)
    scale = thickness / 4

    txt_size = cv2.getTextSize(label, 0, fontScale=scale, thickness=thickness)[0]
    br = (tl[0] + txt_size[0], tl[1] - txt_size[1] - 3)
    mat = cv2.rectangle(mat, tl, br, fill_color, -1, cv2.LINE_AA)  # filled
    return cv2.putText(mat, label, (tl[0], tl[1] - 2), 0, scale, color, thickness=txt_thickness,
                        lineType=cv2.LINE_AA)