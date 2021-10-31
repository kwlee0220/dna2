from __future__ import annotations
from typing import List, Tuple, Union

from omegaconf import OmegaConf
import numpy as np

from dna import Size2d, Box
from .image_capture import ImageCapture
from .default_image_capture import DefaultImageCapture
from .video_file_capture import VideoFileCapture


class Camera:
    def __init__(self, camera_id: str, uri: str, size: Size2d, fps: int, blind_regions: List[Box]) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.size = size
        self.fps = fps
        self.blind_regions = blind_regions
        
    @staticmethod
    def from_conf(conf: OmegaConf) -> Camera:
        size = Size2d.from_np(np.array(conf.size, dtype=np.int32))
        blind_regions = [Box.from_tlbr(np.array(region, dtype=np.int32)) for region in conf.blind_regions]

        return Camera(conf.id, conf.uri, size, conf.fps, blind_regions)
        
    def get_capture(self, sync=True, begin_frame=1, end_frame=None) -> ImageCapture:
        if self.uri.startswith('rtsp://'):
            return DefaultImageCapture(self.uri, target_size=self.size,
                                        begin_frame=begin_frame, end_frame=end_frame)
        elif self.uri.endswith('.mp4') or self.uri.endswith('.avi'):
            return VideoFileCapture(self.uri, target_size=self.size, sync=sync,
                                    begin_frame=begin_frame, end_frame=end_frame)
        elif self.uri.isnumeric():
            return DefaultImageCapture(self.uri, target_size=self.size,
                                        begin_frame=begin_frame, end_frame=end_frame)
        else:
            raise ValueError(f"unknown camera uri='{self.uri}'")
    
    def __repr__(self) -> str:
        return f"{self.camera_id}({self.size}), fps={self.fps}, uri={self.uri}"