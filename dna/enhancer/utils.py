from typing import List, Tuple
from pathlib import Path
import sys

import numpy as np

dna_home = Path(__file__).parents[2].resolve().absolute()
sys.path.append(str(dna_home.absolute()))
from dna import Point


if __name__ == '__main__':
    pt1 = Point(2, 2)
    pt2 = Point(4, 3)

    print(Point.split_points(pt1, pt2, 1))