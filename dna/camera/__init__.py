from .image_capture import ImageCapture
from .default_image_capture import DefaultImageCapture
from .video_file_capture import VideoFileCapture
from .synced_capture import SyncImageCapture
from .image_processor import ImageProcessor
from .utils import load_image_capture

from enum import Enum
class ImageCaptureType(Enum):
    RTSP = 1
    LOCAL_CAMERA = 2
    VIDEO_FILE = 3
    PLATFORM = 4

def image_capture_type(uri):
    if uri.startswith('rstp://'):
        return ImageCaptureType.RTSP
    elif uri.endswith('.mp4') or uri.endswith('.avi'):
        return ImageCaptureType.VIDEO_FILE
    elif uri.isnumeric():
        return ImageCaptureType.LOCAL_CAMERA
    else:
        return ImageCaptureType.PLATFORM