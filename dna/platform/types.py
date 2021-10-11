from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Tuple
from abc import ABCMeta, abstractmethod


class ResourceSet(metaclass=ABCMeta):
    @abstractmethod
    def get(self, key: Tuple):
        pass

    def exists(self, key:Tuple):
        return self.get(key) is not None

    @abstractmethod
    def get_all(self, cond_expr:str=None, offset:int=None, limit:int=None) -> List:
        pass

    @abstractmethod
    def insert(self, rsc) -> None:
        pass

    @abstractmethod
    def remove(self, key: Tuple) -> None:
        pass

    @abstractmethod
    def remove_all(self) -> None:
        pass