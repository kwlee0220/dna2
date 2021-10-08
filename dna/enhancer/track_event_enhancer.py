from typing import List
from dataclasses import dataclass
from threading import Thread

from pubsub import PubSub

from dna.track import Track, TrackState, ObjectTracker
from dna.track.track_callbacks import TrackerCallback
from .types import TrackEvent


_CHANNEL = "track_events"

def _listen(queue, event_consume):
    for entry in queue.listen():
        event = entry['data']
        event_consume(event)

@dataclass(unsafe_hash=True)
class Session:
    state: TrackState
    pendings: List[Track]

class TrackEventEnhancer(TrackerCallback):
    def __init__(self, pubsub: PubSub, camera_id, event_consume) -> None:
        super().__init__()
        self.sessions = {}

        self.camera_id = camera_id
        self.pubsub = pubsub
        self.mqueue = self.pubsub.subscribe(_CHANNEL)
        thread1 = Thread(target=_listen, args=(self.mqueue, event_consume))
        thread1.start()

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None:
        self.sessions.clear()
        self.mqueue.task_done()

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
        return TrackEvent(camera_id=self.camera_id, luid=track.id, location=track.location,
                            frame_index=track.frame_index, ts=track.utc_epoch)