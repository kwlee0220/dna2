from typing import Tuple
from abc import ABCMeta, abstractmethod
from datetime import datetime

import numpy as np

from dna import Size2d


class ImageCapture(metaclass=ABCMeta):
    @abstractmethod
    def open(self) -> None:
        """Opens this ImageCapture.
        """
        pass

    @abstractmethod
    def close(self) -> None:
        """Closes this ImageCapture.
        """
        pass

    @abstractmethod
    def is_open(self) -> bool:
        """Returns whether this is opened or not.

        Returns:
            bool: True if this is opened, False otherwise.
        """
        pass

    @property
    @abstractmethod
    def size(self) -> Size2d:
        """Returns the size of the images that this ImageCapture captures.

        Returns:
            Size2d: (width, height)
        """
        pass

    @property
    @abstractmethod
    def fps(self) -> int:
        """Returns the fps of this ImageCapture.

        Returns:
            int: frames per second.
        """
        pass

    @property
    @abstractmethod
    def frame_index(self) -> int:
        """Returns the total count of images this ImageCapture captures so far.

        Returns:
            int: The number of frames
        """
        pass

    @abstractmethod
    def capture(self) -> Tuple[float, int, np.ndarray]:
        """Captures an OpenCV image.

        Returns:
            Tuple[float, int, np.ndarray]: (timestamp, frame index, frame).
            timestamp: the result of 'time.time()' calls
            frame_index: the index of the captured image
            frame: OpenCV image
        """
        pass