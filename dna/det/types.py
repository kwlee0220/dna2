from __future__ import annotations
from dataclasses import dataclass

import numpy as np

from dna import Box, Size2d
from dna import plot_utils


@dataclass(frozen=True, unsafe_hash=True)
class Detection:
    bbox: Box
    label: str
    score: float

    # @property
    # def tlwh(self) -> np.ndarray:
    #     return self.bbox.tlwh

    # @property
    # def tlbr(self) -> np.ndarray:
    #     return self.bbox.tlbr

    # @property
    # def tl(self) -> np.ndarray:
    #     return self.bbox.tl

    # @property
    # def br(self) -> np.ndarray:
    #     return self.bbox.br

    def __truediv__(self, rhs) -> Detection:
        if isinstance(rhs, Size2d):
            return Detection(self.bbox / rhs, label=self.label, score=self.score)

        raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        return f'{self.label}:{self.bbox},{self.score:.3f}'

    def draw(self, mat, color, label_color=None, show_score=True, line_thickness=2) -> np.ndarray:
        loc = self.bbox
        mat = loc.draw(mat, color=color, line_thickness=line_thickness)
        if label_color:
            msg = f"{self.label}({self.score:.3f})" if show_score else self.label
            mat = plot_utils.draw_label(mat, msg, loc.tl.astype(int), label_color, color, 2)

        return mat