from typing import List
from pathlib import Path
from threading import Thread

import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.types import TrackEvent
from dna.enhancer.trajectory_uploader import TrajectoryUploader
import dna.utils as utils

_SQL_CREATE_TRACK_EVENTS = """
create table track_events (
    camera_id int not null,
    luid bigint not null,
    bbox box not null,
    frame_index bigint not null,
    ts timestamp not null
)
"""

_SQL_CREATE_TRAJECTORIES = """
create table trajectories (
    camera_id int not null,
    luid bigint not null,
    path path not null,
    path_ts timestamp[] not null,
    path_count int not null,
    path_length real not null,
    begin_ts timestamp not null,
    end_ts timestamp not null
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
    conn = pg2.connect(host=args.db_host, port=args.db_port,
                        user=args.db_user, password=args.db_passwd, dbname=args.db_name)
    conn.autocommit = True

    cur = conn.cursor()
    cur.execute('drop table if exists track_events')
    cur.execute('drop table if exists trajectories')
    cur.execute(_SQL_CREATE_TRACK_EVENTS)
    cur.execute(_SQL_CREATE_TRAJECTORIES)
    cur.close()