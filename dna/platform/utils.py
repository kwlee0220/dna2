from dataclasses import dataclass
from typing import List, Union, Tuple

import numpy as np

from dna import Point, BBox


def deserialize_box(box_str: str):
    pts = list(parse_point_list(box_str[1:-1]))
    return BBox.from_points(*pts)

def serialize_box(box: BBox):
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