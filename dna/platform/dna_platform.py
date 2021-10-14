from typing import List, Union

import psycopg2 as pg2
from psycopg2.extras import execute_values

from .types import ResourceSet
from .camera_info import CameraInfoSet
from .local_path import LocalPathSet


class DNAPlatform:
    def __init__(self, host="localhost", port=5432, user="postgres",
                password="dna2021", dbname="dna") -> None:
        self.conn_parms = {'host': host, 'port': port, 'user': user,
                            'password': password, 'dbname': dbname}
        self.conn = None
        self.resource_set_dict = {'camera_infos': CameraInfoSet(self), 'local_paths': LocalPathSet(self)}

    @property
    def connection(self):
        if not self.conn:
            raise ValueError("not connected to DNAPlatform")
        return self.conn

    def connect(self):
        self.conn = pg2.connect(**self.conn_parms)
        return self.conn

    def disconnect(self) -> None:
        self.conn.close()
        self.conn = None

    def get_resource_set_id_all(self) -> List[str]:
        return self.resource_set_dict.keys()

    def get_resource_set(self, id:str) -> ResourceSet:
        rset = self.resource_set_dict.get(id, None)
        if rset is None:
            raise ValueError(f"unknown ResourceSet: {id}")

        return rset
