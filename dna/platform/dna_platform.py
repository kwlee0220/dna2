from __future__ import annotations
from typing import List, Union, Tuple, Any

import psycopg2 as pg2
from psycopg2.extras import execute_values
from omegaconf.omegaconf import OmegaConf

from dna.camera import ImageCapture
from .types import ResourceSet
from .local_path import LocalPathSet


class DNAPlatform:
    def __init__(self, host="localhost", port=5432, user="postgres",
                password="dna2021", dbname="dna") -> None:
        self.conn_parms = {'host': host, 'port': port, 'user': user,
                            'password': password, 'dbname': dbname}
        self.conn = None
        self.resource_set_dict = {'local_paths': LocalPathSet(self)}

    @classmethod
    def load(cls, conf_dict) -> DNAPlatform:
        return DNAPlatform(**conf_dict['db'])

    @classmethod
    def load_from_config(cls, platform_conf: OmegaConf):
        dict = OmegaConf.to_container(platform_conf)
        return DNAPlatform.load(dict)

    def open_db_connection(self):
        return pg2.connect(**self.conn_parms)

    def get_resource_set_id_all(self) -> List[str]:
        return self.resource_set_dict.keys()

    def get_resource_set(self, id:str) -> ResourceSet:
        rset = self.resource_set_dict.get(id, None)
        if rset is None:
            raise ValueError(f"unknown ResourceSet: {id}")

        return rset

    def get_resource(self, rset_id:str, key:Tuple) -> Tuple[ResourceSet,Any]:
        rset = self.get_resource_set(rset_id)
        return rset, rset.get(key)