from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass

import psycopg2 as pg2
from psycopg2.extras import execute_values
from queue import Queue

from dna import Point
import dna.utils as utils
from .types import TrackEvent
from dna.platform import DNAPlatform, Trajectory


class Session:
    def __init__(self, ev: TrackEvent) -> None:
        self.camera_id = ev.camera_id
        self.luid = ev.luid
        self.last_ts = ev.ts
        self.points = [(ev.location.center, ev.ts)]
        self.first_frame = ev.frame_index
        self.last_frame = ev.frame_index
        self.length = 0

    def append(self, ev: TrackEvent) -> None:
        last_tp = self.points[-1]
        pt = ev.location.center

        self.points.append((pt, ev.ts))
        self.last_frame = ev.frame_index
        self.length += Point.distance(last_tp[0], pt)
        self.last_ts = ev.ts

def _build_trajectory(session: Session) -> Trajectory:
    return Trajectory(camera_id=session.camera_id, luid=session.luid,
                        path=session.points, length=session.length,
                        first_frame=session.first_frame, last_frame=session.last_frame)

class TrajectoryUploader:
    def __init__(self, platform:DNAPlatform, mqueue: Queue, batch_size=10,
                    min_path_count=10, max_idle_frames=5*15) -> None:
        self.mqueue = mqueue
        self.trajectories = platform.get_resource_set("trajectories")
        self.sessions = dict()
        self.batch_size = batch_size
        self.min_path_count = min_path_count
        self.max_idle_frames = max_idle_frames
        self.buffer = []

    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            self.sessions[ev.luid] = Session(ev)
        else:
            session.append(ev)

        completeds = [_build_trajectory(sess) \
                        for sess in self.sessions.values()    \
                            if (ev.frame_index - sess.last_frame) > self.max_idle_frames]
        if len(completeds) > 0:
            for trj in completeds:
                self.sessions.pop(trj.luid, None)
                if len(trj.path) >= self.min_path_count:
                    self.buffer.append(trj)

            if len(self.buffer) > self.batch_size:
                self.trajectories.insert_many(self.buffer)
                self.buffer.clear()

    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']
            if event.camera_id is None:
                break
            self.handle_event(event)