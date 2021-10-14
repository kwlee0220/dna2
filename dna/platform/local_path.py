from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

import itertools
from datetime import datetime
import numpy as np
from psycopg2.extras import execute_values

from dna import Point, BBox
from dna.enhancer.types import TrackEvent
from .types import ResourceSet


@dataclass(frozen=True, unsafe_hash=True)
class LocalPath:
    camera_id: str      # 0
    luid: str           # 1
    length: float       # 2
    first_frame: int    # 3
    last_frame: int     # 4
    continuation: bool  # 5, true if this is not the last block. false if this is the last block
    points: List[Point]   # 6

    def serialize(self):
        points_expr =','.join(['({},{})'.format(*np.rint(tp.xy).astype(int)) for tp in self.points])
        points_expr = "[{}]".format(points_expr)

        return (self.camera_id, self.luid, len(self.points), self.length,
                self.first_frame, self.last_frame, self.continuation, points_expr)

    @classmethod
    def deserialize(cls, tup):
        path = [pt for pt in _parse_point_list(tup[7])]
        return LocalPath(camera_id=tup[0], luid=tup[1], points=path, length=tup[3],
                            first_frame=tup[4], last_frame=tup[5], continuation=tup[6])

    @staticmethod
    def concat(paths: List[LocalPath]) -> LocalPath:
        if paths is None or len(paths) == 0:
            return None
        elif len(paths) == 1:
            return paths[0]
        else:
            first = paths[0]
            last = paths[-1]

            concated = []
            for traj in paths:
                concated.extend(traj.points)
            length = sum([Point.distance(pt, next_pt) for pt, next_pt in zip(concated, concated[1:])], 0)

            return LocalPath(first.camera_id, first.luid, length,
                                first.first_frame, last.last_frame, False, concated)

    def __repr__(self) -> str:
        return (f"LocalPath(camera_id={self.camera_id}, luid={self.luid}, "
                f"point_count={len(self.points)}, length={self.length:.1f})")

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

class LocalPathSet(ResourceSet):
    __SQL_GET = """
        select camera_id, luid, point_count, length, first_frame, last_frame, continuation, points
        from local_paths
        where camera_id=%s and luid=%s
        order by first_frame
    """
    __SQL_GET_ALL = """
        select camera_id, luid, point_count, length, first_frame, last_frame, continuation, points
        from local_paths {} {} {}
    """
    __SQL_INSERT = """
        insert into local_paths(camera_id, luid, point_count, length,
                                first_frame, last_frame, continuation, points)
                            values (%s, %s, %s, %s, %s, %s, %s, %s)
    """
    __SQL_INSERT_MANY = """
        insert into local_paths(camera_id, luid, point_count, length,
                                first_frame, last_frame, continuation, points) values %s
    """
    __SQL_REMOVE = "delete from local_paths where camera_id=%s and luid=%s"
    __SQL_REMOVE_ALL = "delete from local_paths"
    __SQL_CREATE = """
        create table local_paths (
            camera_id varchar not null,
            luid bigint not null,
            point_count int not null,
            length real not null,
            first_frame int not null,
            last_frame int not null,
            continuation boolean not null,
            points path not null
        )
    """
    __SQL_CREATE_INDEX = "create index local_path_idx on local_paths(camera_id, luid)"
    __SQL_DROP = "drop table if exists local_paths"

    def __init__(self, platform) -> None:
        super().__init__()

        self.platform = platform

    def create(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_CREATE)
            cur.execute(LocalPathSet.__SQL_CREATE_INDEX)
            conn.commit()

    def drop(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_DROP)
            conn.commit()

    def get(self, key: Tuple[int, int, int], offset=0, limit=None) -> LocalPath:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_GET, key)
            traj = LocalPath.concat([LocalPath.deserialize(tup) for tup in cur])
            conn.commit()

            return traj

    def get_all(self, cond_expr:str, offset:int=None, limit:int=None) -> List[LocalPath]:
        where_clause = f"where {cond_expr}" if cond_expr else ""
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = LocalPathSet.__SQL_GET_ALL.format(where_clause, offset_clause, limit_clause)

        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(sql)
            path_parts = [LocalPath.deserialize(tup) for tup in cur]
            conn.commit()
            groups = itertools.groupby(path_parts, lambda t: t.luid)
            trajs = [LocalPath.concat(list(parts)) for luid, parts in groups]

            return trajs

    def insert(self, traj:LocalPath) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_INSERT, traj.serialize())
            conn.commit()

    def insert_many(self, paths:List[LocalPath]) -> None:
        values = [path.serialize() for path in paths]

        conn = self.platform.connection
        with conn.cursor() as cur:
            execute_values(cur, LocalPathSet.__SQL_INSERT_MANY, values)
            conn.commit()

    def remove(self, key: Tuple[str]) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_REMOVE, key)
            conn.commit()

    def remove_all(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(LocalPathSet.__SQL_REMOVE_ALL)
            conn.commit()