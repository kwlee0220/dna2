from typing import List
from dataclasses import dataclass
from .types import Trajectory

from queue import Queue
from .types import TrackEvent


@dataclass(unsafe_hash=True)
class Session:
    last_ts: int
    trail: List[TrackEvent]


class TrajectoryUploader:
    def __init__(self, mqueue: Queue, max_idle_millis=5*1000) -> None:
        self.mqueue = mqueue
        self.sessions = dict()
        self.max_idle_millis = max_idle_millis

    def handle_event(self, ev: TrackEvent) -> None:
        session = self.sessions.get(ev.luid, None)
        if session is None:
            session = Session(last_ts=-1, trail=[])
            self.sessions[ev.luid] = session
        session.trail.append(ev)
        session.last_ts = max(session.last_ts, ev.ts)

        completeds = [luid for luid, sess in session.items()    \
                        if (ev.ts - sess.last_ts) > self.max_idle_millis]
        for luid in completeds:
            session = self.sessions.pop(luid, None)
            traj = Trajectory.from_track_events(session.trail)

    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']