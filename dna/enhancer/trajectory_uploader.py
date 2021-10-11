from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass

import numpy as np
import psycopg2 as pg2
from psycopg2.extras import execute_values
from queue import Queue

from dna import Point
import dna.utils as utils
from .types import TrackEvent
from dna.platform import DNAPlatform, Trajectory


class Session:
    def __init__(self, camera_id, luid) -> None:
        self.camera_id = camera_id
        self.luid = luid

        self.points = []
        self.first_frame = -1
        self.last_frame = -1
        self.length = 0

    def append(self, ev: TrackEvent) -> None:
        pt = ev.location.center

        if self.first_frame < 0:
            self.first_frame = ev.frame_index
        else:
            self.length += Point.distance(self.points[-1], pt)
        self.points.append(pt)
        self.last_frame = ev.frame_index

def _build_trajectory(session: Session) -> Trajectory:
    return Trajectory(camera_id=session.camera_id, luid=session.luid,
                        path=session.points, length=session.length,
                        first_frame=session.first_frame, last_frame=session.last_frame)

class TrajectoryUploader:
    def __init__(self, platform:DNAPlatform, mqueue: Queue, batch_size=10,
                    min_path_count=10) -> None:
        self.mqueue = mqueue
        self.trajectories = platform.get_resource_set("trajectories")
        self.sessions = dict()
        self.batch_size = batch_size
        self.min_path_count = min_path_count
        self.buffer = []

    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            session = Session(ev.camera_id, ev.luid)
            self.sessions[ev.luid] = session

        if ev.location:
            session.append(ev)
        else:
            session = self.sessions.pop(ev.luid, None)
            if len(session.points) >= self.min_path_count:
                traj = _build_trajectory(session)
                self.buffer.append(traj)
                if len(self.buffer) > self.batch_size:
                    self.trajectories.insert_many(self.buffer)
                    self.buffer.clear()

    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']
            if event.luid is None:
                break
            self.handle_event(event)