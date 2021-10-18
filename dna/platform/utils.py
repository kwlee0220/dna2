from dataclasses import dataclass
from typing import List, Union, Tuple

import numpy as np

from dna import Point, BBox
from dna.camera import ImageCapture


def deserialize_box(box_str: str):
    pts = list(parse_point_list(box_str))
    return BBox.from_points(pts[1], pts[0])

def serialize_box(box: BBox):
    return "(({},{}),({},{}))".format(*box.tlbr.astype(int))

def parse_point_list(pt_list_str) -> List[Point]:
    begin = -1
    for i, c in enumerate(pt_list_str):
        if c == '(':
            begin = i
        elif c == ')' and begin >= 0:
            parts = pt_list_str[begin+1:i].split(',')
            v = np.array(parts)
            v2 = v.astype(float)
            yield Point.from_np(v2)



def load_image_capture_from_platform(platform, camera_id) -> ImageCapture:
    from dna.platform import DNAPlatform, CameraInfo

    platform = DNAPlatform(host=platform.db.host, port=platform.db.port,
                            user=platform.db.user, password=platform.db.passwd,
                            dbname=platform.db.db_name)
    conn = platform.connect()
    try:
        camera_info_set = platform.get_resource_set("camera_infos")
        camera_info :CameraInfo = camera_info_set.get((camera_id,))
        if camera_info is None:
            raise ValueError(f"unknown camera_id: '{camera_id}'")

        return camera_info.load_image_capture()
    finally:
        platform.disconnect()