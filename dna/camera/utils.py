from .image_capture import ImageCapture
from .default_image_capture import DefaultImageCapture
from .video_file_capture import VideoFileCapture


def load_image_capture(uri, size=None, sync=True, begin_frame=1, end_frame=None) -> ImageCapture:
    if uri.startswith('rtsp://'):
        return DefaultImageCapture(uri, target_size=size,
                                    begin_frame=begin_frame, end_frame=end_frame)
    elif uri.endswith('.mp4') or uri.endswith('.avi'):
        return VideoFileCapture(uri, target_size=size, sync=sync,
                                begin_frame=begin_frame, end_frame=end_frame)
    elif uri.isnumeric():
        return DefaultImageCapture(uri, target_size=size,
                                    begin_frame=begin_frame, end_frame=end_frame)
    else:
        raise ValueError(f"unknown camera uri='{uri}'")