import sys
from pathlib import Path

from .detector import LogReadingDetector, ObjectDetector

class DetectorLoader:
    @classmethod
    def load(cls, uri) -> ObjectDetector:
        if not uri:
            raise ValueError(f"detector id is None")

        parts = uri.split(':', 1)
        id = parts[0]
        query = parts[1] if len(parts) == 2 else ""

        try:
            if id == 'yolov4-torch' or id == 'yolov4':
                from .yolov4_torch_detector import load as load_yolov4_torch
                return load_yolov4_torch(query)
            elif id == 'yolov5-torch' or id == 'yolov5':
                from .yolov5_torch_detector import load as load_yolov5_torch
                return load_yolov5_torch(query)
            elif id == 'file':
                det_file = Path(query)
                return LogReadingDetector(det_file)
        except ModuleNotFoundError as exc:
            # raise exc
            print('failure while load a plugin:', exc.msg, file=sys.stderr)
            raise ModuleNotFoundError(f'unable to load a DNA plugin: URI: "{uri}"')

        raise ValueError(f'unknown detector plugin URI: "{uri}"')