from __future__ import annotations
from typing import List, Union
from dataclasses import dataclass
from enum import Enum

import numpy as np
from numpy.lib.arraysetops import isin

@dataclass(frozen=True, unsafe_hash=True)
class Point:
    xy: np.ndarray

    @property
    def x(self):
        return self.xy[0]

    @property
    def y(self):
        return self.xy[1]

    @classmethod
    def distance(cls, pt1:Point, pt2:Point) -> float:
        return np.linalg.norm(pt1.xy - pt2.xy)

    def __add__(self, rhs) -> Point:
        if isinstance(rhs, Point):
            return Point(xy = self.xy + rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point(xy = self.xy + rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(xy = self.xy + np.array(rhs[0:2]))
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(xy = self.xy + np.array([rhs, rhs]))
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __sub__(self, rhs) -> Union[Point,Size2d]:
        if isinstance(rhs, Point):
            return Size2d(xy = self.xy - rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point(xy = self.xy - rhs.xy)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(xy = self.xy - np.array(rhs[0:2]))
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(xy = self.xy - np.array([rhs, rhs]))
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __truediv__(self, rhs) -> Point:
        if isinstance(rhs, Size2d):
            return Point(self.x / rhs.width, self.y / rhs.height)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x / rhs, self.y / rhs)
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        if isinstance(self.xy[0], int):
            return '({},{})'.format(*self.xy)
        else:
            return '({:.1f},{:.1f})'.format(*self.xy)


@dataclass(frozen=True, unsafe_hash=True)
class Size2d:
    wh: np.ndarray

    def __truediv__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d(wh = self.wh / rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d(wh = self.wh / np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        if isinstance(self.wh[0], int):
            return '{}x{}'.format(*self.wh)
        else:
            return '{:.1f}x{:.1f}'.format(*self.wh)


class BBox:
    def __init__(self, tlwh: np.ndarray) -> None:
        self.__tlwh = tlwh
        self.__tlbr = None

    @property
    def tlwh(self) -> np.ndarray:
        return self.__tlwh

    @property
    def tlbr(self) -> np.ndarray:
        if self.__tlbr is None:
            tl = self.tlwh[0:2]
            br = tl + self.tlwh[2:4]
            self.__tlbr = np.hstack([tl, br])
        return self.__tlbr

    @classmethod
    def from_tlbr(cls, tlbr: np.ndarray) -> BBox:
        tl = tlbr[0:2]
        wh = tlbr[2:4] - tl
        return BBox(np.hstack([tl, wh]))

    @property
    def tl(self) -> np.ndarray:
        return self.tlwh[0:2]

    @property
    def wh(self) -> np.ndarray:
        return self.tlwh[2:4]

    @property
    def br(self) -> np.ndarray:
        return self.tlbr[2:4]

    @property
    def top_left(self) -> Point:
        return Point(xy = self.tl)

    @property
    def bottom_right(self) -> Point:
        return Point(xy = self.br)

    @property
    def center(self) -> Point:
        return Point(xy = self.tl + (self.wh / 2))

    @property
    def size(self) -> Size2d:
        return Size2d(wh = self.tlwh[2:4])

    @property
    def width(self) -> Union[int,float]:
        return self.tlwh[2]

    @property
    def height(self) -> Union[int,float]:
        return self.tlwh[3]

    @classmethod
    def distance(cls, bbox1:BBox, bbox2:BBox) -> float:
        tlbr1 = bbox1.tlbr
        tlbr2 = bbox2.tlbr

        delta1 = tlbr1[[0,3]] - tlbr2[[2,1]]
        delta2 = tlbr2[[0,3]] - tlbr2[[2,1]]
        u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
        v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
        dist = np.linalg.norm(np.concatenate([u, v]))
        return dist

    def draw(self, mat, color, line_thickness=2):
        import cv2

        tlbr = self.tlbr.astype(int)
        return cv2.rectangle(mat, tlbr[0:2], tlbr[2:4], color, thickness=line_thickness, lineType=cv2.LINE_AA)

    def __truediv__(self, rhs) -> BBox:
        if isinstance(rhs, Size2d):
            rhs = rhs.wh
        if isinstance(rhs, np.ndarray):
            tlwh = np.hstack([self.tl/rhs, self.wh/rhs])
            return BBox(tlwh)

        raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self):
        return '{}:{}'.format(self.top_left, self.size)