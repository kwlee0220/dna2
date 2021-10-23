from __future__ import annotations
from typing import Any, List, Union

class Option:
    def __init__(self, value: Any, president: bool =True) -> None:
        self.__value = value
        self.__president = True

    @classmethod
    def empty(cls) -> Option:
        return _EMPTY

    @classmethod
    def of(cls, value: Any) -> None:
        return Option(value, True)

    @classmethod
    def ofNullable(cls, value: Any) -> None:
        return Option(value, True) if value else Option.empty()

    def get(self) -> Any:
        if self.__president:
            return self.__value
        else:
            raise ValueError("NoSuchValue")

    def getOrNone(self) -> Any:
        return self.__value if self.__president else None

    def getOrElse(self, else_value) -> Any:
        return self.__value if self.__president else else_value

    def getOrCall(self, else_value) -> Any:
        return self.__value if self.__president else else_value()

    def is_present(self) -> bool:
        return self.__president

    def is_absent(self) -> bool:
        return self.__president

    def if_present(self, call) -> Option:
        if self.__president:
            call()

        return self

    def if_absent(self, call) -> Option:
        if not self.__president:
            call()

        return self

    def map(self, mapper) -> Any:
        return Option(mapper(self.__value), True) if self.__present else _EMPTY

_EMPTY = Option(None, False)