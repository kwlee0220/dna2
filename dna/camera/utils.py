from .image_capture import ImageCapture
from .default_image_capture import DefaultImageCapture
from .video_file_capture import VideoFileCapture


def load_image_capture(uri, size=None, sync=True) -> ImageCapture:
    if uri.startswith('rstp://'):
        return DefaultImageCapture(uri, target_size=size)
    elif uri.endswith('.mp4') or uri.endswith('.avi'):
        return VideoFileCapture(uri, target_size=size, sync=sync)
    elif uri.isnumeric():
        return DefaultImageCapture(uri, target_size=size)
    else:
        raise ValueError(f"unknown camera uri='{uri}'")