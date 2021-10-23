from __future__ import annotations
from os import stat
from typing import List, Union, Tuple
from dataclasses import dataclass
from enum import Enum

import numpy as np


class Point:
    def __init__(self, x: Union[int, float], y: Union[int, float]) -> None:
        self.__xy = np.array([x, y])

    @property
    def x(self):
        return self.__xy[0]

    @property
    def y(self):
        return self.__xy[1]

    @property
    def xy(self):
        return self.__xy

    @classmethod
    def from_np(cls, xy: np.ndarray) -> Point:
        return Point(xy[0], xy[1])

    def distance_to(self, pt:Point) -> float:
        return np.linalg.norm(self.xy - pt.xy)

    @staticmethod
    def line_function(pt1:Point, pt2:Point):
        delta = pt1.xy - pt2.xy
        if delta[0] == 0:
            raise ValueError(f"Cannot find a line function: {pt1} - {pt}")
        slope = delta[1] / delta[0]
        y_int = pt2.y - (slope * pt2.x)

        def func(x):
            return (slope * x) + y_int
        return func

    @staticmethod
    def split_points(pt1: Point, pt2: Point, npoints: int) -> List[Point]:
        func = Point.line_function(pt1, pt2)
        step_x = (pt2.x - pt1.x) / (npoints+1)
        xs = [pt1.x + (idx * step_x) for idx in range(1, npoints+1)]
        return [Point.from_np(np.array([x, func(x)])) for x in xs]

    def __add__(self, rhs) -> Point:
        if isinstance(rhs, Point):
            return Point.from_np(self.xy + rhs.xy)
        elif isinstance(rhs, Size2d):
            return Point.from_np(self.xy + rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(self.x + rhs[0], self.y + rhs[1])
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x + rhs, self.y + rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __sub__(self, rhs) -> Union[Point,Size2d]:
        if isinstance(rhs, Point):
            return Size2d.from_np(self.xy - rhs.xy)
        elif isinstance(rhs, Size2d) or isinstance(rhs, Size2i):
            return Point.from_np(self.xy - rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(self.x - rhs[0], self.y - rhs[1])
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x - rhs, self.y - rhs)
        else:
            raise ValueError(f"invalid rhs: rhs={rhs}")

    def __mul__(self, rhs) -> Point:
        if isinstance(rhs, int) or isinstance(rhs, float):
            return Point(self.x * rhs, self.y * rhs)
        elif isinstance(rhs, Size2d) or isinstance(rhs, Size2i):
            return Point.from_np(self.xy * rhs.wh)
        elif isinstance(rhs, tuple) and len(rhs) >= 2:
            return Point(self.x * rhs[0], self.y * rhs[1])
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


class Size2d:
    def __init__(self, width: Union[int, float], height: Union[int, float]) -> None:
        self.__wh = np.array([width, height])

    @classmethod
    def from_np(cls, wh: np.ndarray) -> Point:
        return Size2d(wh[0], wh[1])

    def is_valid(self) -> bool:
        return self.__wh[0] >= 0 and self.__wh[1] >= 0

    @property
    def wh(self) -> np.ndarray:
        return self.__wh

    @property
    def width(self) -> float:
        return self.__wh[0]
    
    @property
    def height(self) -> float:
        return self.__wh[1]

    def area(self) -> float:
        return self.__wh[0] * self.__wh[1]

    def abs(self) -> Size2d:
        return Size2d.from_np(np.abs(self.__wh))

    def to_size2i(self):
        return Size2i(np.rint(self.wh).astype(int))

    def __sub__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d) or isinstance(rhs, Size2i):
            return Size2d.from_np(self.wh - rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh - np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __mul__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh * rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh * np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __truediv__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2d):
            return Size2d.from_np(self.wh / rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.wh / np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        if isinstance(self.wh[0], int):
            return '{}x{}'.format(*self.__wh)
        else:
            return '{:.1f}x{:.1f}'.format(*self.__wh)
EMPTY_SIZE2D: Size2d = Size2d(-1, -1)


class Size2i:
    def __init__(self, width: int, height: int) -> None:
        self.__wh = np.array([width, height])

    @classmethod
    def from_np(cls, wh: np.ndarray) -> Size2i:
        return Size2i(int(wh[0]), int(wh[1]))

    @property
    def wh(self) -> np.ndarray:
        return self.__wh

    @property
    def width(self) -> int:
        return int(self.__wh[0])
    
    @property
    def height(self) -> int:
        return int(self.__wh[1])

    def area(self) -> int:
        return self.__wh[0] * self.__wh[1]

    def as_tuple(self) -> Tuple[int,int]:
        return tuple(self.__wh)

    def __mul__(self, rhs) -> Size2i:
        if isinstance(rhs, Size2i):
            return Size2i.from_np(self.__wh / rhs.wh)
        elif isinstance(rhs, int):
            return Size2i(self.width*rhs, self.height*rhs)
        elif isinstance(rhs, float):
            return Size2i.from_np(self.__wh * rhs)
        else:
            raise ValueError('invalid right-hand-side:', rhs)

    def __truediv__(self, rhs) -> Size2d:
        if isinstance(rhs, Size2i) or isinstance(rhs, Size2d):
            return Size2d.from_np(self.__wh / rhs.wh)
        elif isinstance(rhs, int) or isinstance(rhs, float):
            return Size2d.from_np(self.__wh / np.array([rhs, rhs]))
        else:
            raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self) -> str:
        return '{}x{}'.format(*self.__wh)


class Box:
    def __init__(self, tl: np.ndarray, br: np.ndarray, wh: np.ndarray) -> None:
        self.__tl = tl
        self.__br = br
        self.__wh = wh

    @classmethod
    def from_points(self, tl: Point, br: Point) -> Box:
        return Box(tl.xy, br.xy, br.xy - tl.xy)

    @classmethod
    def from_tlbr(cls, tlbr: np.ndarray) -> Box:
        tl = tlbr[:2]
        br = tlbr[2:]

        return Box(tl, br, br - tl)

    @classmethod
    def from_tlwh(cls, tlwh: np.ndarray) -> Box:
        tl = tlwh[0:2]
        wh = tlwh[2:4]
        return Box(tl, tl + wh, wh)

    def is_valid(self) -> bool:
        return self.__wh[0] >= 0 and self.__wh[1] >= 0

    @property
    def tlbr(self) -> np.ndarray:
        return np.hstack([self.__tl, self.__br])

    @property
    def tlwh(self) -> np.ndarray:
        return np.hstack([self.__tl, self.__wh])

    @property
    def tl(self) -> np.ndarray:
        return self.__tl

    @property
    def br(self) -> np.ndarray:
        return self.__br

    @property
    def wh(self) -> np.ndarray:
        return self.__wh

    @property
    def top_left(self) -> Point:
        return Point.from_np(self.__tl)

    @property
    def bottom_right(self) -> Point:
        return Point.from_np(self.__br)

    def center(self) -> Point:
        return Point.from_np(self.__tl + (self.wh / 2.))

    def size(self) -> Size2d:
        return Size2d.from_np(self.__wh) if self.is_valid() else EMPTY_SIZE2D

    def area(self) -> int:
        return self.size().area() if self.is_valid() else 0

    def width(self) -> Union[int,float]:
        return self.size().width

    def height(self) -> Union[int,float]:
        return self.size().height

    def distance_to(self, bbox:Box) -> float:
        tlbr1 = self.tlbr
        tlbr2 = bbox.tlbr

        delta1 = tlbr1[[0,3]] - tlbr2[[2,1]]
        delta2 = tlbr2[[0,3]] - tlbr2[[2,1]]
        u = np.max(np.array([np.zeros(len(delta1)), delta1]), axis=0)
        v = np.max(np.array([np.zeros(len(delta2)), delta2]), axis=0)
        dist = np.linalg.norm(np.concatenate([u, v]))
        return dist

    def intersection(self, bbox: Box) -> Union[Box, None]:
        x1 = max(self.tl[0], bbox.tl[0])
        y1 = max(self.tl[1], bbox.tl[1])
        x2 = min(self.br[0], bbox.br[0])
        y2 = min(self.br[1], bbox.br[1])
        
        if x1 >= x2 or y1 >= y2:
            return EMPTY_BBox
        else:
            return Box.from_tlbr(np.array([x1, y1, x2, y2]))

    def contains(self, box: Box) -> bool:
        return self.tlbr[0] <= box.tlbr[0] and self.tlbr[1] <= box.tlbr[1] \
                and self.tlbr[2] >= box.tlbr[2] and self.tlbr[3] <= box.tlbr[3]

    def draw(self, mat, color, line_thickness=2):
        import cv2

        tlbr = self.tlbr.astype(int)
        return cv2.rectangle(mat, tlbr[0:2], tlbr[2:4], color, thickness=line_thickness, lineType=cv2.LINE_AA)

    def __truediv__(self, rhs) -> Box:
        if isinstance(rhs, Size2d):
            wh = self.size().wh / rhs
            return Box.from_tlwh(self.tl, wh)

        raise ValueError('invalid right-hand-side:', rhs)
    
    def __repr__(self):
        return '{}:{}'.format(self.top_left, self.size())

EMPTY_BBox: Box = Box(np.array([-1,-1]), np.array([0,0]), np.array([-1,-1]))