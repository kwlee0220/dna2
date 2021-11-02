from typing import Union
from datetime import datetime

import numpy as np
from psycopg2.extras import execute_values
from queue import Queue
from omegaconf import OmegaConf
from shapely import wkb
from shapely.geometry import Point

from dna.platform import DNAPlatform
from .types import TrackEvent
from .event_processor import EventProcessor


_INSERT_SQL = "insert into track_events(camera_id, luid, bbox, world_coord, distance, frame_index, ts) values %s"
class TrackEventUploader(EventProcessor):
    def __init__(self, platform: DNAPlatform, in_queue: Queue, conf: OmegaConf) -> None:
        self.in_queue = in_queue
        self.bulk = []
        self.batch_size = conf.batch_size
        self.conn = platform.open_db_connection()

    def close(self) -> None:
        if len(self.bulk) >= 0:
            self.__upload()
        self.bulk.clear()
        self.conn.close()

    def handle_event(self, ev: TrackEvent) -> None:
        if ev.location:
            self.bulk.append(self.__serialize_event(ev))
            if len(self.bulk) >= self.batch_size:
                self.__upload()

    def subscribe(self) -> Union[Queue, None]:
        return None

    def __upload(self):
        with self.conn.cursor() as cur:
            # execute_values(cur, _INSERT_SQL, self.bulk)
            execute_values(cur, _INSERT_SQL, self.bulk)
            self.conn.commit()
        self.bulk.clear()

    def __serialize_event(self, ev: TrackEvent):
        box_expr = '({},{}),({},{})'.format(*np.rint(ev.location.tlbr).astype(int))
        if ev.world_coord is not None:
            world_coord = ev.world_coord.wkb_hex
        else:
            world_coord = None
        return (ev.camera_id, ev.luid, box_expr, world_coord, ev.distance,
                ev.frame_index, datetime.fromtimestamp(ev.ts))