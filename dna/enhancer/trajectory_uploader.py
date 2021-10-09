from datetime import datetime, timedelta
from typing import List
from dataclasses import dataclass
from .types import Trajectory

import psycopg2 as pg2
from psycopg2.extras import execute_values
from queue import Queue

import dna.utils as utils
from .types import TrackEvent, TimedPoint, Trajectory


@dataclass(unsafe_hash=True)
class Session:
    last_ts: datetime
    trail: List[TrackEvent]

    def append(self, ev: TrackEvent) -> None:
        self.trail.append(ev)
        self.last_ts = max(self.last_ts, ev.ts)


_INSERT_SQL = "insert into trajectories(camera_id, luid, path, path_ts, path_count, path_length, begin_ts, end_ts) values %s"

def _encode(traj: Trajectory):
    path_expr = ','.join(['({},{})'.format(*tp.location.center.xy) for tp in traj.path])
    path_ts_expr = [tp.ts for tp in traj.path]

    return (traj.camera_id, traj.luid, path_expr, path_ts_expr, len(traj.path), traj.length, traj.begin_ts, traj.end_ts)

class TrajectoryUploader:
    def __init__(self, mqueue: Queue, conn, max_idle_millis=timedelta(seconds=5)) -> None:
        self.mqueue = mqueue
        self.conn = conn
        self.sessions = dict()
        self.max_idle_millis = max_idle_millis
        self.buffer = []

    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            session = Session(last_ts=datetime.min, trail=[])
            self.sessions[ev.luid] = session
        session.append(ev)

        completeds = [luid for luid, sess in self.sessions.items()    \
                        if (ev.ts - sess.last_ts) > self.max_idle_millis]
        for luid in completeds:
            session = self.sessions.pop(luid, None)
            traj = Trajectory.from_track_events(session.trail)
            
            self.buffer.append(_encode(traj))
            if len(self.buffer) >= 30:
                cur= self.conn.cursor()
                execute_values(cur, _INSERT_SQL, self.buffer)
                self.conn.commit()
                cur.close()
                self.buffer.clear()

    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']
            if event.camera_id is None:
                break
            self.handle_event(event)