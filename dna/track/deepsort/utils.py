from typing import List, Union, Tuple

import numpy as np


def boxes_distance(tlbr1, tlbr2):
    delta1 = tlbr1[0,3] - tlbr2[2,1]
    delta2 = tlbr2[0,3] - tlbr2[2,1]
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist

def overlap_ratio(box1, box2) -> Tuple[float,float]:
    inter_area = box1.intersection(box2).area()
    r1 = inter_area / box1.area() if box1.is_valid() else 0
    r2 = inter_area / box2.area() if box2.is_valid() else 0
    return max(r1, r2)

def find_overlaps(box, candidate_boxes, threshold, candidate_indices=None) -> List[Tuple[int,float]]:
    if not candidate_indices:
        candidate_indices = list(range(len(candidate_boxes)))

    overlaps = []
    for cidx in candidate_indices:
        ratio = overlap_ratio(box, candidate_boxes[cidx])
        if ratio > threshold:
            overlaps.append((cidx, ratio))

    return overlaps

def split_tuples(tuples: List[Tuple]):
    firsts = []
    seconds = []
    for t in tuples:
        firsts.append(t[0])
        seconds.append(t[1])

    return firsts, seconds

def project(tuples: List[Tuple], idx: int):
    return [t[idx] for t in tuples]