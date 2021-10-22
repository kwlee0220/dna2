from __future__ import annotations
from dataclasses import dataclass
from typing import List, Tuple, Union

import numpy as np
from psycopg2.extras import execute_values

from dna import Size2i, Box, Point
from dna.camera import ImageCapture, DefaultImageCapture, VideoFileCapture
from dna.platform import ResourceSet, utils


class CameraInfo:
    def __init__(self, camera_id, uri, size, fps, blind_regions :List[Box]=[]) -> None:
        self.camera_id = camera_id
        self.uri = uri
        self.size = size
        self.fps = fps
        self.blind_regions = blind_regions

    def add_blind_region(self, region: Box) -> CameraInfo:
        self.blind_regions.append(region)
        return self

    def serialize(self) -> Tuple:
        return (self.camera_id, str(self.uri), self.size.width, self.size.height, self.fps)

    @classmethod
    def deserialize(cls, tup: Tuple, blind_regions=None) -> CameraInfo:
        size = Size2i(*tup[2:4])
        return CameraInfo(camera_id=tup[0], uri=tup[1], size=size, fps=tup[4],
                            blind_regions=blind_regions)
    
    def __repr__(self) -> str:
        return f"{self.camera_id}({self.size}), fps={self.fps}, uri={self.uri}"


class CameraInfoSet(ResourceSet):
    __SQL_GET = "select camera_id, uri, width, height, fps from cameras where camera_id=%s"
    __SQL_GET_REGIONS = "select region from blind_regions where camera_id=%s"
    __SQL_GET_ALL = "select camera_id, uri, width, height, fps from cameras {} {} {}"
    __SQL_INSERT = "insert into cameras(camera_id, uri, width, height, fps) values (%s, %s, %s, %s, %s)"
    __SQL_INSERT_BLIND_REGION = "insert into blind_regions(camera_id,region) values %s"
    __SQL_REMOVE = "delete from cameras where camera_id=%s"
    __SQL_REMOVE_BLINKD_REGION = "delete from blind_regions where camera_id=%s"
    __SQL_REMOVE_ALL = "delete from cameras"
    __SQL_CREATE = """
        create table cameras (
            camera_id varchar not null,
            uri varchar not null,
            width int not null,
            height int not null,
            fps int not null,

            constraint cameras_pkey primary key (camera_id)
        )
    """
    __SQL_CREATE_BLIND_REGION = """
        create table blind_regions (
            camera_id varchar not null,
            region box not null
        )
    """
    __SQL_DROP = "drop table if exists cameras"
    __SQL_DROP_BLIND_REGION = "drop table if exists blind_regions"

    def __init__(self, platform) -> None:
        super().__init__()

        self.platform = platform

    def create(self) -> None:
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_CREATE)
                cur.execute(CameraInfoSet.__SQL_CREATE_BLIND_REGION)
                conn.commit()

    def drop(self) -> None:
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_DROP)
                cur.execute(CameraInfoSet.__SQL_DROP_BLIND_REGION)
                conn.commit()

    def get(self, key: Tuple[str]) -> CameraInfo:
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_GET, key)
                tup = cur.fetchone()

                cur.execute(CameraInfoSet.__SQL_GET_REGIONS, key)
                regions = [utils.deserialize_box(tup[0]) for tup in cur]
                conn.commit()

                return CameraInfo.deserialize(tup, regions) if tup else None

    def get_all(self, cond_expr:str=None, offset:int=None, limit:int=None) -> List[CameraInfo]:
        where_clause = f"where {cond_expr}" if cond_expr else ""
        offset_clause = f"offset {offset}" if offset else ""
        limit_clause = f"limit {offset}" if limit else ""
        sql = CameraInfoSet.__SQL_GET_ALL.format(where_clause, offset_clause, limit_clause)
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(sql)
                trajs = [CameraInfoSet.deserialize(tup) for tup in cur]
                conn.commit()

                return trajs

    def insert(self, camera_info:CameraInfo) -> None:
        regions = [(camera_info.camera_id, utils.serialize_box(region)) for region in camera_info.blind_regions]

        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_INSERT, camera_info.serialize())
                if len(camera_info.blind_regions) > 0:
                    execute_values(cur, CameraInfoSet.__SQL_INSERT_BLIND_REGION, regions)
                conn.commit()

    def update_blind_regions(self, camera_info:CameraInfo) -> None:
        boxes = [(camera_info.camera_id, utils.serialize_box(region)) for region in camera_info.blind_regions]

        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_REMOVE_BLINKD_REGION, (camera_info.camera_id,))
                if len(camera_info.blind_regions) > 0:
                    execute_values(cur, CameraInfoSet.__SQL_INSERT_BLIND_REGION, boxes)
                conn.commit()

    def remove(self, key: Tuple[str]) -> None:
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_REMOVE, key)
                conn.commit()

    def remove_all(self) -> None:
        with self.platform.open_db_connection() as conn:
            with conn.cursor() as cur:
                cur.execute(CameraInfoSet.__SQL_REMOVE_ALL)
                conn.commit()