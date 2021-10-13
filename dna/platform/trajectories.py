from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import itertools
import functools
from datetime import datetime
import numpy as np
from psycopg2.extras import execute_values

from dna import Point, BBox
from dna.enhancer.types import TrackEvent
from .types import ResourceSet


@dataclass(frozen=True, unsafe_hash=True)
class Trajectory:
    camera_id: str      # 0
    luid: str           # 1
    length: float       # 2
    first_frame: int    # 3
    last_frame: int     # 4
    continuation: bool  # 5, true if this is not the last block. false if this is the last block
    path: List[Point]   # 6

    def serialize(self):
        path_expr =','.join(['({},{})'.format(*np.rint(tp.xy).astype(int)) for tp in self.path])
        path_expr = "[{}]".format(path_expr)

        return (self.camera_id, self.luid, len(self.path), self.length,
                self.first_frame, self.last_frame, self.continuation, path_expr)

    @classmethod
    def deserialize(cls, tup):
        path = [pt for pt in _parse_point_list(tup[7])]
        return Trajectory(camera_id=tup[0], luid=tup[1], path=path, length=tup[3],
                            first_frame=tup[4], last_frame=tup[5], continuation=tup[6])

    @staticmethod
    def concat(trajs: List[Trajectory]) -> Trajectory:
        if trajs is None or len(trajs) == 0:
            return None
        elif len(trajs) == 1:
            return trajs[0]
        else:
            first = trajs[0]
            last = trajs[-1]

            path = []
            for traj in trajs:
                path.extend(traj.path)
            length = sum([Point.distance(pt1, pt2) for pt1, pt2 in zip(path, path[1:])], 0)

            return Trajectory(first.camera_id, first.luid, length, first.first_frame, last.last_frame, False, path)

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
        select camera_id, luid, path_count, path_length, first_frame, last_frame, continuation, path
        from trajectories
        where camera_id=%s and luid=%s
        order by first_frame
    """
    __SQL_GET_ALL = """
        select camera_id, luid, path_count, path_length, first_frame, last_frame, continuation, path
        from trajectories {} {} {}
    """
    __SQL_INSERT = """
        insert into trajectories(camera_id, luid, path_count, path_length,
                                first_frame, last_frame, continuation, path)
                            values (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    __SQL_INSERT_MANY = """
        insert into trajectories(camera_id, luid, path_count, path_length,
                                first_frame, last_frame, continuation, path) values %s
    """
    __SQL_REMOVE = "delete from trajectories where camera_id=%s and luid=%s"
    __SQL_REMOVE_ALL = "delete from trajectories"
    __SQL_CREATE = """
        create table trajectories (
            camera_id varchar not null,
            luid bigint not null,
            path_count int not null,
            path_length real not null,
            first_frame int not null,
            last_frame int not null,
            continuation boolean not null,
            path path not null
        )
    """
    __SQL_CREATE_INDEX = "create index traj_idx on trajectories(camera_id, luid)"
    __SQL_DROP = "drop table if exists trajectories"

    def __init__(self, platform) -> None:
        super().__init__()

        self.platform = platform

    def create(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_CREATE)
            cur.execute(TrajectorySet.__SQL_CREATE_INDEX)
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
            traj = Trajectory.concat([Trajectory.deserialize(tup) for tup in cur])
            conn.commit()

            return traj

    def get_all(self, cond_expr:str, offset:int=None, limit:int=None) -> List[Trajectory]:
        where_clause = f"where {cond_expr}" if cond_expr else ""
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = TrajectorySet.__SQL_GET_ALL.format(where_clause, offset_clause, limit_clause)

        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(sql)
            traj_parts = [Trajectory.deserialize(tup) for tup in cur]
            conn.commit()
            groups = itertools.groupby(traj_parts, lambda t: t.luid)
            trajs = [Trajectory.concat(list(parts)) for luid, parts in groups]

            return trajs

    def insert(self, traj:Trajectory) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(TrajectorySet.__SQL_INSERT, traj.serialize())
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