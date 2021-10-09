from typing import List
from pathlib import Path
from threading import Thread

from pubsub import PubSub, Queue
import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import VideoFileCapture
from dna.det import DetectorLoader
from dna.track import DeepSORTTracker, ObjectTrackingProcessor
from dna.enhancer import TrackEventEnhancer
from dna.enhancer.types import TrackEvent
from dna.enhancer.trajectory_uploader import TrajectoryUploader
import dna.utils as utils


_INSERT_SQL = "insert into track_events(camera_id,luid,bbox,frame_index,ts) values %s"
_BULK_SIZE = 100

def store_track_event(conn, mqueue: Queue):
    print("starting a thread...")

    bulk = []
    for entry in mqueue.listen():
        ev = entry['data']
        if ev.camera_id is None:
            if len(bulk) > 0:
                _upload(conn, bulk)
            break

        bulk.append(_to_values(ev))
        if len(bulk) >= _BULK_SIZE:
            _upload(conn, bulk)

def _upload(conn, bulk):
    cur = conn.cursor()
    execute_values(cur, _INSERT_SQL, bulk)
    conn.commit()
    cur.close()
    bulk.clear()


def _to_values(ev: TrackEvent):
    box_expr = '({},{}),({},{})'.format(*ev.location.tlbr)
    return (ev.camera_id, ev.luid, box_expr, ev.frame_index, ev.ts)

import argparse
def parse_args():
    parser = argparse.ArgumentParser(description="Detect Objects in a video file")
    parser.add_argument("--home", help="DNA framework home directory.", default=".")
    parser.add_argument("--camera_id", help="camera id")
    parser.add_argument("--detector", help="Object detection algorithm.", default="yolov4")
    parser.add_argument("--match_score", help="Mathing threshold", default=0.55)
    parser.add_argument("--max_iou_distance", help="maximum IoU distance", default=0.99)
    parser.add_argument("--max_age", type=int, help="max. # of frames to delete", default=30)
    parser.add_argument("--input", help="input source.", required=True)
    parser.add_argument("--show", help="show detections.", action="store_true")

    parser.add_argument("--db_host", help="host name of DNA data platform", default="localhost")
    parser.add_argument("--db_port", type=int, help="port number of DNA data platform", default=5432)
    parser.add_argument("--db_name", help="database name", default="dna")
    parser.add_argument("--db_user", help="user name", default="postgres")
    parser.add_argument("--db_passwd", help="password", default="dna2021")
    return parser.parse_args()


import dna.utils as utils

if __name__ == '__main__':
    args = parse_args()

    capture = VideoFileCapture(Path(args.input))
    detector = DetectorLoader.load(args.detector)

    dna_home_dir = Path(args.home)
    model_file = dna_home_dir / 'dna' / 'track' / 'deepsort' / 'ckpts' / 'model640.pt'
    tracker = DeepSORTTracker(detector, weights_file=model_file.absolute(),
                                matching_threshold=args.match_score,
                                max_iou_distance=args.max_iou_distance,
                                max_age=args.max_age)

    pubsub = PubSub()
    enhancer = TrackEventEnhancer(pubsub, args.camera_id)

    conn = pg2.connect(host=args.db_host, port=args.db_port,
                        user=args.db_user, password=args.db_passwd, dbname=args.db_name)

    thread = Thread(target=store_track_event, args=(conn, enhancer.subscribe(),))
    thread.start()

    trj_upload = TrajectoryUploader(enhancer.subscribe(), conn)
    thread = Thread(target=trj_upload.run, args=tuple())
    thread.start()

    win_name = "output" if args.show else None
    with ObjectTrackingProcessor(capture, tracker, enhancer, window_name=win_name) as processor:
        from timeit import default_timer as timer
        from datetime import timedelta

        started = timer()
        frame_count = processor.run()
        elapsed = timer() - started
        fps = frame_count / elapsed

    print(f"elapsed_time={timedelta(seconds=elapsed)}, fps={fps:.1f}" )