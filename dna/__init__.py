from .types import Point, Size2d, Size2i, BBox
from pathlib import Path
from .utils import get_logger

import os
DNA_HOME = Path(os.environ.get('DNA_HOME', '.'))
DNA_CONIFIG_FILE = DNA_HOME / 'conf' / 'config.yaml'

DEBUG_FRAME_IDX = -1
DEBUG_SHOW_IMAGE = False
DEBUG_PRINT_COST = False
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


# from dna.camera import ImageCapture, DefaultImageCapture, VideoFileCapture
# def load_image_capture(uri, sync=True, size=None,
#                         begin_frame=1, end_frame=None, platform=None) -> ImageCapture:
#     if not uri.startswith('rstp://') \
#         and not uri.endswith('.mp4') and not uri.endswith('.avi') \
#         and not uri.isnumeric():
#         if platform:
#             _, camera_info = platform.get_resource("camera_infos", (uri,))
#             uri = camera_info.uri
#         else:
#             raise ValueError(f"unknown camera uri='{uri}'")

#     if uri.startswith('rstp://'):
#         return DefaultImageCapture(uri, target_size=size,
#                                     begin_frame=begin_frame, end_frame=end_frame)
#     elif uri.endswith('.mp4') or uri.endswith('.avi'):
#         return VideoFileCapture(uri, target_size=size, sync=sync,
#                                 begin_frame=begin_frame, end_frame=end_frame)
#     elif uri.isnumeric():
#         return DefaultImageCapture(uri, target_size=size,
#                                     begin_frame=begin_frame, end_frame=end_frame)
#     else:
#         raise ValueError(f"unknown camera uri='{uri}'")