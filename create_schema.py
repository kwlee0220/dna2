from typing import List
from pathlib import Path
from threading import Thread

import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import  Size2i
from dna.platform import CameraInfo, DNAPlatform

_SQL_CREATE_TRACK_EVENTS = """
create table track_events (
    camera_id varchar not null,
    luid bigint not null,
    bbox box not null,
    frame_index bigint not null,
    ts timestamp not null
)
"""

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")

    parser.add_argument("--db_host", help="host name of DNA data platform", default="localhost")
    parser.add_argument("--db_port", type=int, help="port number of DNA data platform", default=5432)
    parser.add_argument("--db_name", help="database name", default="dna")
    parser.add_argument("--db_user", help="user name", default="postgres")
    parser.add_argument("--db_passwd", help="password", default="dna2021")
    parser.add_argument("--drop_if_exists", help="drop tables if exist", action="store_true")
    return parser.parse_args()

if __name__ == '__main__':
    args = parse_args()

    dna_home_dir = Path(args.home)
    platform = DNAPlatform(host=args.db_host, port=args.db_port,
                            user=args.db_user, password=args.db_passwd, dbname=args.db_name)
    conn = platform.connect()

    cur = conn.cursor()
    cur.execute('drop table if exists track_events')
    conn.commit()
    cur.close()

    for id in platform.get_resource_set_id_all():
        rset = platform.get_resource_set(id)
        rset.drop()
        rset.create()

    camera_infos = platform.get_resource_set("camera_infos")
    camera_infos.insert(CameraInfo(camera_id='ai_city:1', size=Size2i(1280, 960), fps=10))
    camera_infos.insert(CameraInfo(camera_id='ai_city:6', size=Size2i(1280, 960), fps=10))
    camera_infos.insert(CameraInfo(camera_id='ai_city:9', size=Size2i(1920, 1080), fps=10))
    camera_infos.insert(CameraInfo(camera_id='ai_city:11', size=Size2i(1920, 1080), fps=10))
    camera_infos.insert(CameraInfo(camera_id='etri:5', size=Size2i(1920, 1080), fps=10))
    camera_infos.insert(CameraInfo(camera_id='etri:6', size=Size2i(1920, 1080), fps=10))
    camera_infos.insert(CameraInfo(camera_id='test:1', size=Size2i(1920, 1080), fps=10))

    cur = conn.cursor()
    cur.execute(_SQL_CREATE_TRACK_EVENTS)
    conn.commit()
    cur.close()
    conn.close()