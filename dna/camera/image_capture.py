from typing import Tuple
from abc import ABCMeta, abstractmethod
from datetime import datetime

import numpy as np

from dna import Size2i



class ImageCapture(metaclass=ABCMeta):
    @abstractmethod
    def is_open(self) -> bool:
        pass

    @abstractmethod
    def open(self) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    @abstractmethod
    def is_open(self) -> bool:
        pass

    @property
    @abstractmethod
    def size(self) -> Size2i:
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        pass

    @abstractmethod
    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        pass