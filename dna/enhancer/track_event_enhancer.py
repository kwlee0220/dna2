from typing import List
from dataclasses import dataclass
from threading import Thread

from pubsub import PubSub, Queue

from dna import BBox
from dna.track import Track, TrackState, ObjectTracker
from dna.track.track_callbacks import TrackerCallback
from .types import TrackEvent


@dataclass(unsafe_hash=True)
class Session:
    state: TrackState
    pendings: List[Track]

_CHANNEL = "track_events"
class TrackEventEnhancer(TrackerCallback):
    def __init__(self, pubsub: PubSub, camera_id) -> None:
        super().__init__()
        self.sessions = {}

        self.camera_id = camera_id
        self.pubsub = pubsub
        self.queue = self.pubsub.subscribe(_CHANNEL)

    def subscribe(self) -> Queue:
        return self.pubsub.subscribe(_CHANNEL)

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None:
        self.pubsub.publish(_CHANNEL, TrackEvent(camera_id=None, luid=None,
                                                location=None, frame_index=None, ts=None))
        self.sessions.clear()
        self.queue.task_done()

    def tracked(self, tracker: ObjectTracker, frame, frame_idx: int, tracks: List[Track]) -> None:
        for track in tracks:
            if track.state == TrackState.Deleted:
                session = self.sessions.pop(track.id, None)
                print(f"drop {len(session.pendings)} events")
                continue

            session = self.sessions.get(track.id, None)
            if session is None: # TrackState.Null or TrackState.Deleted
                if track.state == TrackState.Confirmed:
                    self.sessions[track.id] = Session(track.state, [])
                elif track.state == TrackState.Tentative:
                    self.sessions[track.id] = Session(track.state, [track])
            elif session.state == TrackState.Tentative:
                if track.state == TrackState.Confirmed:
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
        self.pubsub.publish(_CHANNEL, self.__to_event(track))

    def __to_event(self, track):
        location = BBox.from_tlbr(track.location.tlbr.astype(int))
        return TrackEvent(camera_id=self.camera_id, luid=track.id, location=location,
                            frame_index=track.frame_index, ts=track.ts)