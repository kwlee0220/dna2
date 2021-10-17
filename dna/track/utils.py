from typing import List
from cv2 import line

import numpy as np

from dna import BBox, plot_utils
from . import Track, TrackState


def draw_track_trail(mat, track: Track, color, label_color=None,
                    trail: List[BBox]=None, trail_color=None, line_thickness=2) -> np.ndarray:
    mat = track.draw(mat, color, label_color=label_color, line_thickness=2)
    if trail_color:
        mat = plot_utils.draw_line_string(mat, [bbox.center() for bbox in trail[-11:]],
                                        trail_color, line_thickness)
    return mat