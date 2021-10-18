from __future__ import annotations
from typing import List, Union, Tuple, Any

import psycopg2 as pg2
from psycopg2.extras import execute_values

from dna.camera.image_capture import ImageCapture

from .types import ResourceSet
from .camera_info import CameraInfo, CameraInfoSet
from .local_path import LocalPathSet


class DNAPlatform:
    def __init__(self, host="localhost", port=5432, user="postgres",
                password="dna2021", dbname="dna") -> None:
        self.conn_parms = {'host': host, 'port': port, 'user': user,
                            'password': password, 'dbname': dbname}
        self.conn = None
        self.resource_set_dict = {'camera_infos': CameraInfoSet(self), 'local_paths': LocalPathSet(self)}

    @classmethod
    def load(cls, conf_dict) -> DNAPlatform:
        return DNAPlatform(**conf_dict['db'])

    def open_db_connection(self):
        return pg2.connect(**self.conn_parms)

    # @property
    # def connection(self):
    #     if not self.conn:
    #         raise ValueError("not connected to DNAPlatform")
    #     return self.conn

    # def connect(self):
    #     self.conn = pg2.connect(**self.conn_parms)
    #     return self.conn

    # def disconnect(self) -> None:
    #     self.conn.close()
    #     self.conn = None

    def get_resource_set_id_all(self) -> List[str]:
        return self.resource_set_dict.keys()

    def get_resource_set(self, id:str) -> ResourceSet:
        rset = self.resource_set_dict.get(id, None)
        if rset is None:
            raise ValueError(f"unknown ResourceSet: {id}")

        return rset

    def get_resource(self, rset_id:str, key:Tuple) -> Any:
        rset = self.get_resource_set(rset_id)
        return rset.get(key)

    def load_image_capture(self, camera_id: str, sync=True) -> ImageCapture:
        camera_info_set = self.get_resource_set("camera_infos")
        camera_info = self.get_resource("camera_infos", (camera_id,))
        if camera_info is None:
            raise ValueError(f"unknown camera_id: '{camera_id}'")

        import dna.camera.utils as camera_utils
        return camera_utils.load_image_capture(camera_info.uri, camera_info.size, sync=sync)