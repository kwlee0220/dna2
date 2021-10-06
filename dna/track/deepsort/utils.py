import numpy as np


def boxes_distance(tlbr1, tlbr2):
    delta1 = tlbr1[0,3] - tlbr2[2,1]
    delta2 = tlbr2[0,3] - tlbr2[2,1]
    u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
    v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
    dist = np.linalg.norm(np.concatenate([u, v]))
    return dist