from typing import List
from dataclasses import dataclass
from pathlib import Path
from threading import Thread

from pubsub import PubSub

from dna import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import Track, TrackState, ObjectTracker, DeepSORTTracker, ObjectTrackingProcessor
from dna.track.track_callbacks import TrackerCallback


_CHANNEL = "track_events"

def _listen(cam_id, queue, event_consume):
    for entry in queue.listen():
        track = entry['data']
        event_consume(cam_id, track.id, track.location, track.frame_index)

    print('XXXXXXXXXXXXXXXX')

@dataclass(unsafe_hash=True)
class Session:
    state: TrackState
    pendings: List[Track]

class TrackEventEnhancer(TrackerCallback):
    def __init__(self, camera_id, event_consume) -> None:
        super().__init__()
        self.sessions = {}

        self.pubsub = PubSub()

        self.mqueue = self.pubsub.subscribe(_CHANNEL)
        thread1 = Thread(target=_listen, args=(camera_id, self.mqueue, event_consume))
        thread1.start()

    def track_started(self, tracker: ObjectTracker) -> None: pass
    def track_stopped(self, tracker: ObjectTracker) -> None:
        self.mqueue.unsubscribe()

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
                    for pended in session.pendings:
                        confirmed = self.__to_confirmed_track(pended)
                        self.pubsub.publish(_CHANNEL, confirmed)
                    self.pubsub.publish(_CHANNEL, track)
                    session.state = TrackState.Confirmed
                    session.pendings.clear()
                elif track.state == TrackState.Tentative:
                    session.pendings.append(track)
            elif session.state == TrackState.Confirmed:
                if track.state == TrackState.Confirmed:
                    self.pubsub.publish(_CHANNEL, track)
                elif track.state == TrackState.TemporarilyLost:
                    session.pendings.append(track)
                    session.state = TrackState.TemporarilyLost
            elif session.state == TrackState.TemporarilyLost:
                if track.state == TrackState.Confirmed:
                    for pended in session.pendings:
                        confirmed = self.__to_confirmed_track(pended)
                        self.pubsub.publish(_CHANNEL, confirmed)
                    self.pubsub.publish(_CHANNEL, track)
                    session.state = TrackState.Confirmed
                    session.pendings.clear()
                elif track.state == TrackState.TemporarilyLost:
                    session.pendings.append(track)

    def __to_confirmed_track(self, track):
        return Track(track.id, TrackState.Confirmed, track.location, track.frame_index)