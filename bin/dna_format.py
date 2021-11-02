from typing import List
from pathlib import Path
from threading import Thread

import numpy as np
import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import  DNA_CONIFIG_FILE, parse_config_args, load_config
from dna.platform import DNAPlatform


_SQL_CREATE_TRACK_EVENTS = """
create table track_events (
    camera_id varchar not null,
    luid bigint not null,
    bbox box not null,
    world_coord geometry(pointz),
    distance real,
    frame_index bigint not null,
    ts timestamp not null
)
"""

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--conf", help="DNA framework configuration", default=DNA_CONIFIG_FILE)

    parser.add_argument("--drop_if_exists", help="drop tables if exist", action="store_true")
    return parser.parse_known_args()

if __name__ == '__main__':
    args, unknown = parse_args()
    config_grp = parse_config_args(unknown)

    conf = load_config(DNA_CONIFIG_FILE)
    platform = DNAPlatform.load_from_config(conf.platform)

    with platform.open_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('drop table if exists track_events')
            conn.commit()

    for id in platform.get_resource_set_id_all():
        rset = platform.get_resource_set(id)
        rset.drop()
        rset.create()

    cur = conn.cursor()
    cur.execute(_SQL_CREATE_TRACK_EVENTS)
    conn.commit()
    cur.close()
    conn.close()