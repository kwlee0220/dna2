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
    
    info = CameraInfo(camera_id='ai_city:01', uri="C:/Temp/data/ai_city/cam_01.mp4", size=Size2i(1280, 960), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([7,2,504,115])))
    info.add_blind_region(Box.from_tlbr(np.array([10,591,279,816])))
    info.add_blind_region(Box.from_tlbr(np.array([909,38,1063,138])))
    info.add_blind_region(Box.from_tlbr(np.array([1045,76,1154,155])))
    info.add_blind_region(Box.from_tlbr(np.array([1156,6,1283,287])))
    camera_infos.insert(info)
    
    info = CameraInfo(camera_id='ai_city:06', uri="C:/Temp/data/ai_city/cam_06.mp4", size=Size2i(1280, 960), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([234,2,1054,145])))
    info.add_blind_region(Box.from_tlbr(np.array([806,126,1171,229])))
    info.add_blind_region(Box.from_tlbr(np.array([-8,-1,245,179])))
    info.add_blind_region(Box.from_tlbr(np.array([1237,30,1279,373])))
    info.add_blind_region(Box.from_tlbr(np.array([1164,66,1247,295])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='ai_city:09', uri="C:/Temp/data/ai_city/cam_09.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-13,1,1219,228])))
    info.add_blind_region(Box.from_tlbr(np.array([1209,-6,1435,159])))
    info.add_blind_region(Box.from_tlbr(np.array([1425,1,1918,110])))
    info.add_blind_region(Box.from_tlbr(np.array([1237,30,1279,373])))
    info.add_blind_region(Box.from_tlbr(np.array([1725,100,1917,172])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='ai_city:11', uri="C:/Temp/data/ai_city/cam_11.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-6,-1,979,189])))
    info.add_blind_region(Box.from_tlbr(np.array([1412,6,1930,271])))
    camera_infos.insert(info)
                                    
    info = CameraInfo(camera_id='etri:05', uri="C:/Temp/data/etri/etri_05.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([-17,-3,727,391])))
    info.add_blind_region(Box.from_tlbr(np.array([719,29,1904,249])))
    info.add_blind_region(Box.from_tlbr(np.array([1307,232,1915,398])))
    info.add_blind_region(Box.from_tlbr(np.array([-5,828,1235,1082])))
    camera_infos.insert(info)

    info = CameraInfo(camera_id='etri:06', uri="C:/Temp/data/etri/etri_06.mp4", size=Size2i(1920, 1080), fps=10)
    info.add_blind_region(Box.from_tlbr(np.array([1309,223,1908,488])))
    info.add_blind_region(Box.from_tlbr(np.array([743,87,1081,316])))
    info.add_blind_region(Box.from_tlbr(np.array([1741,508,1923,1087])))
    info.add_blind_region(Box.from_tlbr(np.array([-3,634,387,1083])))
    camera_infos.insert(info)
                                    
    info = CameraInfo(camera_id='etri_live:04',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/3/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info)  
    info = CameraInfo(camera_id='etri_live:05',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/4/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info) 
    info = CameraInfo(camera_id='etri_live:06',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/5/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info) 
    info = CameraInfo(camera_id='etri_live:07',
                            uri="rtsp://admin:dnabased24@129.254.82.33:558/LiveChannel/6/media.smp",
                            size=Size2i(1280, 720), fps=10)
    camera_infos.insert(info)

    info = CameraInfo(camera_id='cross:01', uri="C:/Temp/data/crossroads/cross_01.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([502,-9,1568,175])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:02', uri="C:/Temp/data/crossroads/cross_02.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([464,-2,1479,144])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:03', uri="C:/Temp/data/crossroads/cross_03.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([559,-2,1609,58])))
    info.add_blind_region(Box.from_tlbr(np.array([1,878,1925,1084])))
    info.add_blind_region(Box.from_tlbr(np.array([1483,457,1925,1086])))
    info.add_blind_region(Box.from_tlbr(np.array([1698,222,1939,705])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:04', uri="C:/Temp/data/crossroads/cross_04.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([654,-2,1505,68])))
    info.add_blind_region(Box.from_tlbr(np.array([655,-2,1357,67])))
    info.add_blind_region(Box.from_tlbr(np.array([1375,60,1530,195])))
    info.add_blind_region(Box.from_tlbr(np.array([1438,579,1923,1085])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:11', uri="C:/Temp/data/crossroads/cross_11.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([501,-3,1573,236])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:12', uri="C:/Temp/data/crossroads/cross_12.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([464,-2,1479,144])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:13', uri="C:/Temp/data/crossroads/cross_13.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([559,-2,1604,58])))
    camera_infos.insert(info)
    info = CameraInfo(camera_id='cross:21', uri="C:/Temp/data/crossroads/cross_21.mp4", size=Size2i(1920, 1080), fps=15)
    info.add_blind_region(Box.from_tlbr(np.array([501,-3,1573,236])))
    camera_infos.insert(info)

    camera_infos.insert(CameraInfo(camera_id='test', uri="C:/Temp/local_path.mp4",
                                    size=Size2i(1920, 1080), fps=10))

    cur = conn.cursor()
    cur.execute(_SQL_CREATE_TRACK_EVENTS)
    conn.commit()
    cur.close()
    conn.close()