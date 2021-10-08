import psycopg2 as pg2
from psycopg2.extras import execute_values

from .types import TrackEvent

_INSERT_SQL = "insert into track_events(camera_id,luid,bbox,frame_index,ts) values %s"

class UploadEventBulk:
    def __init__(self, conn, bulk_size: int) -> None:
        self.bulk_size = bulk_size
        self.bulk = []

        self.cur = conn.cursor()
        
    def upload(self, ev: TrackEvent) -> None:
        self.bulk.append(ev)
        if len(self.bulk) >= self.bulk_size:
            values = [self.__tuple(ev) for ev in self.bulk]
            execute_values(self.cur, _INSERT_SQL, values)
            self.bulk.clear()

    def __tuple(self, ev):
        return (ev.camera_id, ev.luid, (tuple(ev.location.tl), tuple(ev.location.br)), ev.frame_index, ev.ts)