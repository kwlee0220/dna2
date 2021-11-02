from typing import List
from dataclasses import dataclass
from threading import Thread
import logging

from pubsub import PubSub, Queue
from omegaconf import OmegaConf

from dna import Box, get_logger
from dna.track import Track, TrackState, ObjectTracker
from dna.track.track_callbacks import TrackerCallback
from .types import TrackEvent, end_of_track_event


_CHANNEL = "track_events"
@dataclass(unsafe_hash=True)
class Session:
    state: TrackState
    pendings: List[Track]

class TrackEventEnhancer(TrackerCallback):
    def __init__(self, pubsub: PubSub, camera_id, conf: OmegaConf) -> None:
        super().__init__()
        self.sessions = {}

        self.camera_id = camera_id
        self.pubsub = pubsub
        self.queue = self.pubsub.subscribe(_CHANNEL)

        self.logger = get_logger("dna_enhancer")
        level_name = conf.get("log_level", "info").upper()
        level = logging.getLevelName(level_name)
        self.logger.setLevel(level)

    def subscribe(self) -> Queue:
        return self.pubsub.subscribe(_CHANNEL)

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None:
        stop = end_of_track_event(self.camera_id)
        self.pubsub.publish(_CHANNEL, stop)
        self.sessions.clear()
        self.queue.task_done()

    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            if track.state == TrackState.Deleted:
                session = self.sessions.pop(track.id, None)
                if session.state == TrackState.Tentative:
                    self.logger.debug(f"drop immature track[{track.id}]")
                else:
                    self.__publish(track)
                    count = len(session.pendings)
                    if count > 0:
                        self.logger.debug(f"drop trailing tracks[{track.id}]: count={count}")

                continue

            session = self.sessions.get(track.id, None)
            if session is None: # TrackState.Null or TrackState.Deleted
                if track.state == TrackState.Confirmed:
                    self.sessions[track.id] = Session(track.state, [])
                elif track.state == TrackState.Tentative:
                    self.sessions[track.id] = Session(track.state, [track])
            elif session.state == TrackState.Tentative:
                if track.state == TrackState.Confirmed:
                    # self.logger.debug(f"accept tentative tracks: track={track.id}, count={len(session.pendings)}")
                    self.__publish_all_pended_events(session)
                    self.__publish(track)
                    session.state = TrackState.Confirmed
                elif track.state == TrackState.Tentative:
                    session.pendings.append(track)
            elif session.state == TrackState.Confirmed:
                if track.state == TrackState.Confirmed:
                    self.__publish(track)
                elif track.state == TrackState.TemporarilyLost:
                    session.pendings.append(track)
                    session.state = TrackState.TemporarilyLost
            elif session.state == TrackState.TemporarilyLost:
                if track.state == TrackState.Confirmed:
                    self.logger.debug(f"accept 'temporarily-lost' tracks[{track.id}]: count={len(session.pendings)}")

                    self.__publish_all_pended_events(session)
                    self.__publish(track)
                    session.state = TrackState.Confirmed
                elif track.state == TrackState.TemporarilyLost:
                    session.pendings.append(track)

    def __publish_all_pended_events(self, session):
        for pended in session.pendings:
            self.__publish(pended)
        session.pendings.clear()

    def __publish(self, track):
        location = None if track.is_deleted() else track.location
        ev = TrackEvent(camera_id=self.camera_id, luid=track.id, location=location,
                            world_coord=None, distance=None,
                            frame_index=track.frame_index, ts=track.ts)
        self.pubsub.publish(_CHANNEL, ev)