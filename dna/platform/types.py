from __future__ import annotations
from dataclasses import dataclass
from typing import List, Union, Tuple
from abc import ABCMeta, abstractmethod


class ResourceSet(metaclass=ABCMeta):
    @abstractmethod
    def get(self, key: Tuple):
        pass

    @abstractmethod
    def get_all(self) -> List:
        pass

    @abstractmethod
    def get_where(self, cond_expr:str, offset:int=None, limit:int=None) -> List:
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