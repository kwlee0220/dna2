from typing import List
from pathlib import Path
from threading import Thread

import numpy as np
import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna import  Size2i
from dna.platform import CameraInfo, DNAPlatform
from dna.types import Box

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
    with platform.open_db_connection() as conn:
        with conn.cursor() as cur:
            cur.execute('drop table if exists track_events')
            cur.execute('drop table if exists blind_regions')
            conn.commit()

    for id in platform.get_resource_set_id_all():
        rset = platform.get_resource_set(id)
        rset.drop()
        rset.create()

    camera_infos = platform.get_resource_set("camera_infos")
    
    info = CameraInfo(camera_id='ai_city:1', uri="C:/Temp/data/cam_1.mp4",
                                    size=Size2i(1280, 960), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([7,2,504,115])))
    info.add_blind_region(Box.from_tlbr(np.array([10,591,279,816])))
    info.add_blind_region(Box.from_tlbr(np.array([909,38,1063,138])))
    info.add_blind_region(Box.from_tlbr(np.array([1045,76,1154,155])))
    camera_infos.insert(info)
    
    info = CameraInfo(camera_id='ai_city:6', uri="C:/Temp/data/cam_6.mp4", size=Size2i(1280, 960), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([234,2,1054,145])))
    info.add_blind_region(Box.from_tlbr(np.array([806,126,1171,229])))
    info.add_blind_region(Box.from_tlbr(np.array([-8,-1,245,179])))
    info.add_blind_region(Box.from_tlbr(np.array([1237,30,1279,373])))
    info.add_blind_region(Box.from_tlbr(np.array([1164,66,1247,295])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='ai_city:9', uri="C:/Temp/data/cam_9.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-13,1,1219,228])))
    info.add_blind_region(Box.from_tlbr(np.array([1209,-6,1435,159])))
    info.add_blind_region(Box.from_tlbr(np.array([1425,1,1918,110])))
    info.add_blind_region(Box.from_tlbr(np.array([1237,30,1279,373])))
    info.add_blind_region(Box.from_tlbr(np.array([1725,100,1917,172])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='ai_city:11', uri="C:/Temp/data/cam_11.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-6,-1,979,189])))
    info.add_blind_region(Box.from_tlbr(np.array([1412,6,1930,271])))
    camera_infos.insert(info)
                                    
    info = CameraInfo(camera_id='etri:5', uri="C:/Temp/data/etri_5.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-17,-3,727,391])))
    info.add_blind_region(Box.from_tlbr(np.array([719,29,1904,249])))
    info.add_blind_region(Box.from_tlbr(np.array([1307,232,1915,398])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='etri:6', uri="C:/Temp/data/etri_6.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([1309,223,1908,488])))
    info.add_blind_region(Box.from_tlbr(np.array([743,87,1081,316])))
    camera_infos.insert(info)
                                    
    info = CameraInfo(camera_id='etri_live:3',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/3/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info)  
    info = CameraInfo(camera_id='etri_live:4',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/4/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info) 
    info = CameraInfo(camera_id='etri_live:5',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/5/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info) 
    info = CameraInfo(camera_id='etri_live:6',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/6/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info)

    camera_infos.insert(CameraInfo(camera_id='test', uri="C:/Temp/local_path.mp4",
                                    size=Size2i(1920, 1080), fps=10))

    cur = conn.cursor()
    cur.execute(_SQL_CREATE_TRACK_EVENTS)
    conn.commit()
    cur.close()
    conn.close()