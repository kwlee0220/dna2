
import time
from datetime import datetime
import numpy as np
from psycopg2.extras import execute_values
from queue import Queue

from dna.platform import DNAPlatform
from .types import TrackEvent


_INSERT_SQL = "insert into track_events(camera_id, luid, bbox, frame_index, ts) values %s"
class TrackEventUploader:
    def __init__(self, platform:DNAPlatform, mqueue: Queue, bulk_size:int=100) -> None:
        self.mqueue = mqueue
        self.bulk = []
        self.bulk_size = bulk_size
        self.conn = platform.open_db_connection()

    def handle_event(self, ev: TrackEvent) -> None:
        self.bulk.append(self.__serialize(ev))
        if len(self.bulk) >= self.bulk_size:
            self.__upload()
    
    def run(self) -> None:
        for entry in self.mqueue.listen():
            event = entry['data']
            if event.luid is None:
                break
            elif event.location:
                self.handle_event(event)

        if len(self.bulk) >= 0:
            self.__upload()

    def __upload(self):
        with self.conn.cursor() as cur:
            execute_values(cur, _INSERT_SQL, self.bulk)
            self.conn.commit()
        self.bulk.clear()

    def __serialize(self, ev: TrackEvent):
        box_expr = '({},{}),({},{})'.format(*np.rint(ev.location.tlbr).astype(int))
        return (ev.camera_id, ev.luid, box_expr, ev.frame_index, datetime.fromtimestamp(ev.ts))