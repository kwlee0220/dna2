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
    def capture(self) -> Tuple[datetime, int, np.ndarray]:
        pass

    @abstractmethod
    def size(self) -> Size2i:
        pass

    @property
    @abstractmethod
    def frame_count(self) -> int:
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        pass

    @property
    @abstractmethod
    def fps(self) -> float:
        pass

    def __enter__(self):
        if self.is_open():
            raise ValueError(f"{self.__class__.__name__}: invalid state (opened already)")
        self.open()

        return self

    def __exit__(self, exc_type, exc_value, exc_tb):
        if self.is_open():
            self.close()

    def __repr__(self) -> str:
        state = 'opened' if self.is_open() else 'closed'
        return (f"{__class__.__name__}[({str(self.size)}):{state}]: "
                f"frames={self.frame_index}/{self.frame_count}, "
                f"fps={self.fps:.0f}/s")