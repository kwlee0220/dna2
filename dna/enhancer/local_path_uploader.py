from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass

import numpy as np
import psycopg2 as pg2
from psycopg2.extras import execute_values
from queue import Queue
from omegaconf import OmegaConf
from shapely.geometry import LineString

from dna import Point, get_logger
import dna.utils as utils
from .types import TrackEvent
from dna.platform import DNAPlatform, LocalPath

_MAX_BLOCK_SIZE = 1000
_logger = get_logger('dna.enhancer')

class Session:
    def __init__(self, camera_id, luid) -> None:
        self.camera_id = camera_id
        self.luid = luid

        self.points = []
        self.world_coords = []
        self.first_frame = -1
        self.last_frame = -1
        self.length = 0

    def is_too_long(self) -> bool:
        return len(self.points) >= _MAX_BLOCK_SIZE

    def append(self, ev: TrackEvent) -> None:
        pt = ev.location.center()

        if self.first_frame < 0:
            self.first_frame = ev.frame_index
        else:
            self.length += pt.distance_to(self.points[-1])
        self.points.append(pt)
        self.world_coords.append(ev.world_coord)
        self.last_frame = ev.frame_index

    def build_local_path(self, cont: TrackEvent=None) -> LocalPath:
        if cont:
            self.world_coords.append(cont.world_coord)
        return LocalPath(camera_id=self.camera_id, luid=self.luid,
                            points=self.points, length=self.length,
                            line=LineString(self.world_coords),
                            first_frame=self.first_frame, last_frame=self.last_frame,
                            continuation=cont is not None)

class LocalPathUploader:
    def __init__(self, platform:DNAPlatform, mqueue: Queue, conf: OmegaConf) -> None:
        self.mqueue = mqueue
        self.local_paths = platform.get_resource_set("local_paths")
        self.sessions = dict()
        self.batch_size = conf.batch_size
        self.min_path_count = conf.min_path_count
        self.max_pending_sec = timedelta(seconds=conf.max_pending_sec)
        self.local_path_buffer = []
        self.last_upload_ts = datetime.now()

    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']
            if event.luid is None:
                # build local paths from the unfinished sessions and upload them
                for session in self.sessions.values():
                    if len(session.points) >= self.min_path_count:
                        path = session.build_local_path()
                        self.upload(path)
                self.flush()

                self.sessions.clear()
                break

            self.handle_event(event)

    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            session = Session(ev.camera_id, ev.luid)
            self.sessions[ev.luid] = session

        if ev.location: # ordinary TrackEvent
            if session.is_too_long():   # if session keeps too many points
                path = session.build_local_path(ev)
                self.upload(path)

                # refresh the current session
                session = Session(ev.camera_id, ev.luid)
                self.sessions[ev.luid] = session
            session.append(ev)
        else:   # end of track
            self.sessions.pop(ev.luid, None)

            # ignore this local path if it has too few points
            if len(session.points) >= self.min_path_count:
                path = session.build_local_path()
                self.upload(path)

    def upload(self, path: LocalPath, force: bool=False):
        self.local_path_buffer.append(path)

        npendings = len(self.local_path_buffer)
        if not force and npendings < self.batch_size \
            and (datetime.now() - self.last_upload_ts) < self.max_pending_sec:
            return

        self.flush()

    def flush(self):
        self.local_paths.insert_many(self.local_path_buffer)
        self.last_upload_ts = datetime.now()
        _logger.info(f"upload {len(self.local_path_buffer)} local_paths")
        self.local_path_buffer.clear()