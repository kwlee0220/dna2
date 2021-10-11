from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple

from dna import Size2i
from dna.platform import ResourceSet

@dataclass(frozen=True, unsafe_hash=True)
class CameraInfo:
    camera_id: str
    size: Size2i

    def serialize(self) -> Tuple:
        return (self.camera_id, self.size.width, self.size.height)

    @classmethod
    def deserialize(cls, tup: Tuple) -> CameraInfo:
        size = Size2i(list(tup[1:3]))
        return CameraInfo(camera_id=tup[0], size=size)
    
    def __repr__(self) -> str:
        return f"{self.camera_id}({self.size})"


class CameraInfoSet(ResourceSet):
    __SQL_GET = "select camera_id, width, height from cameras where camera_id=%s"
    __SQL_GET_ALL = "select camera_id, width, height from cameras {} {} {}"
    __SQL_INSERT = "insert into cameras(camera_id, width, height) values (%s, %s, %s)"
    __SQL_REMOVE = "delete from cameras where camera_id=%s"
    __SQL_REMOVE_ALL = "delete from cameras"
    __SQL_CREATE = """
        create table cameras (
            camera_id varchar not null,
            width int not null,
            height int not null,

            constraint cameras_pkey primary key (camera_id)
        )
    """
    __SQL_DROP = "drop table if exists cameras"

    def __init__(self, platform) -> None:
        super().__init__()

        self.platform = platform

    def create(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_CREATE)
            conn.commit()

    def drop(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_DROP)
            conn.commit()

    def get(self, key: Tuple[str]) -> CameraInfo:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_GET, key)
            tup = cur.fetchone()
            conn.commit()

            return CameraInfo.deserialize(tup) if tup else None

    def get_all(self, cond_expr:str=None, offset:int=None, limit:int=None) -> List[CameraInfo]:
        where_clause = f"where {cond_expr}" if cond_expr else ""
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = CameraInfoSet.__SQL_GET_ALL.format(where_clause, offset_clause, limit_clause)
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(sql)
            trajs = [CameraInfoSet.deserialize(tup) for tup in cur]
            conn.commit()

            return trajs

    def insert(self, camera_info:CameraInfo) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_INSERT, camera_info.serialize())
            conn.commit()

    def remove(self, key: Tuple[str]) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_REMOVE, key)
            conn.commit()

    def remove_all(self) -> None:
        conn = self.platform.connection
        with conn.cursor() as cur:
            cur.execute(CameraInfoSet.__SQL_REMOVE_ALL)
            conn.commit()