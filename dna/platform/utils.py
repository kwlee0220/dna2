from dataclasses import dataclass
from typing import List, Union, Tuple

import numpy as np

from dna import Point, Box
from dna.camera import ImageCapture


def deserialize_box(box_str: str):
    pts = list(parse_point_list(box_str))
    return Box.from_points(pts[1], pts[0])

def serialize_box(box: Box):
    return "(({},{}),({},{}))".format(*box.tlbr.astype(int))

def parse_point_list(pt_list_str) -> List[Point]:
    begin = -1
    for i, c in enumerate(pt_list_str):
        if c == '(':
            begin = i
        elif c == ')' and begin >= 0:
            parts = pt_list_str[begin+1:i].split(',')
            v = np.array(parts)
            v2 = v.astype(float)
            yield Point.from_np(v2)