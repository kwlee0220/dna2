from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from datetime import datetime
from psycopg2.extras import execute_values
import numpy as np

from dna import Point, BBox
from dna.enhancer.types import TrackEvent
from .types import ResourceSet


@dataclass(frozen=True, unsafe_hash=True)
class Trajectory:
    camera_id: str
    luid: str
    path: List[Tuple[Point,datetime]]
    length: float
    first_frame: int
    last_frame: int

    def serialize(self):
        path_expr =','.join(['({},{})'.format(*tp[0].xy) for tp in self.path])
        path_expr = "[{}]".format(path_expr)
        path_ts_expr = [tp[1] for tp in self.path]

        return (self.camera_id, self.luid, len(self.path), self.length,
                self.first_frame, self.last_frame, path_expr, path_ts_expr, )

    @classmethod
    def deserialize(cls, tup):
        points = [pt for pt in _parse_point_list(tup[6])]
        path = [(pt, ts) for pt, ts in zip(points, tup[7])]
        return Trajectory(camera_id=tup[0], luid=tup[1], path=path,
                            length=tup[3], first_frame=tup[4], last_frame=tup[5])

    def __repr__(self) -> str:
        return (f"Trajectory(camera_id={self.camera_id}, luid={self.luid}, "
                f"npoints={len(self.path)}, length={self.length:.1f})")

def _parse_point_list(path_str):
    begin = -1
    for i, c in enumerate(path_str):
        if c == '(':
            begin = i
        elif c == ')' and begin >= 0:
            parts = path_str[begin+1:i].split(',')
            v = np.array(parts)
            v2 = v.astype(float)
            yield Point(v2)

class TrajectorySet(ResourceSet):
    __SQL_GET = """
        select camera_id, luid, path_count, path_length, first_frame, last_frame, path, path_ts
        from trajectories
        where camera_id=%s and luid=%s and first_frame=%s
    """
    __SQL_GET_ALL = """
        select camera_id, luid, path_count, path_length, first_frame, last_frame, path, path_ts
        from trajectories {} {}
    """
    __SQL_GET_WHERE = """
        select camera_id, luid, path_count, path_length, first_frame, last_frame, path, path_ts
        from trajectories where {} {} {}
    """
    __SQL_INSERT = """
        insert into trajectories(camera_id, luid, path_count, path_length,
                                first_frame, last_frame, path, path_ts)
                            values (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    __SQL_INSERT_MANY = """
        insert into trajectories(camera_id, luid, path_count, path_length,
                                first_frame, last_frame, path, path_ts) values %s
    """
    __SQL_REMOVE = "delete from trajectories where camera_id=%s and luid=%s and first_frame=%s"
    __SQL_REMOVE_ALL = "delete from trajectories"
    __SQL_CREATE = """
        create table trajectories (
            camera_id varchar not null,
            luid bigint not null,
            path_count int not null,
            path_length real not null,
            first_frame int not null,
            last_frame int not null,
            path path not null,
            path_ts timestamp[] not null
        )
    """
    __SQL_DROP = "drop table if exists trajectories"

    def __init__(self, platform) -> None:
        super().__init__()

        self.platform = platform

    def create(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_CREATE)
            conn.commit()

    def drop(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_DROP)
            conn.commit()

    def get(self, key: Tuple[int, int, int], offset=0, limit=None) -> Trajectory:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_GET, key)
            tup = cur.fetchone()
            conn.commit()

            return Trajectory.deserialize(tup) if tup else None

    def get_where(self, cond_expr:str, offset:int=None, limit:int=None) -> List[Trajectory]:
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = TrajectorySet.__SQL_GET_WHERE.format(cond_expr, offset_clause, limit_clause)
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(sql)
            trajs = [Trajectory.deserialize(tup) for tup in cur]
            conn.commit()

            return trajs

    def get_all(self, offset:int=None, limit:int=None) -> List[Trajectory]:
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = TrajectorySet.__SQL_GET_ALL.format(offset_clause, limit_clause)

        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(sql)
            trajs = [Trajectory.deserialize(tup) for tup in cur]
            conn.commit()

            return trajs

    def insert(self, traj:Trajectory) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_INSERT, camera_info.serialize())
            conn.commit()

    def insert_many(self, trajs:List[Trajectory]) -> None:
        values = [traj.serialize() for traj in trajs]

        conn = self.platform.connection
        with conn.cursor() as cur:
            execute_values(cur, TrajectorySet.__SQL_INSERT_MANY, values)
            conn.commit()

    def remove(self, key: Tuple[str]) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_REMOVE, key)
            conn.commit()

    def remove_all(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_REMOVE_ALL)
            conn.commit()