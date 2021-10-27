from .types import Point, Size2d, Size2i, Box
from pathlib import Path
from .utils import get_logger

import os
DNA_HOME = Path(os.environ.get('DNA_HOME', '.'))
DNA_CONIFIG_FILE = DNA_HOME / 'conf' / 'config.yaml'

DEBUG_FRAME_IDX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = False
DEBUG_START_FRAME = 1164
DEBUG_TARGET_TRACKS = None

from collections import defaultdict
def parse_config_args(args): 
    config_grp = defaultdict(list) 
    for arg in args:
        idx = arg.find('=')
        if idx >= 0:
            key = arg[:idx]
            value = arg[idx+1:]
            idx = key.find('.')
            grp_key = key[:idx]
            key = key[idx+1:]
            config_grp[grp_key].append((key, value))
    return config_grp