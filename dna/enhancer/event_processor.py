from typing import Union
from abc import ABCMeta, abstractmethod

from queue import Queue


class EventProcessor(metaclass=ABCMeta):
    @abstractmethod
    def subscribe(self) -> Union[Queue, None]:
        pass

    @abstractmethod
    def handle_event(self, ev) -> None:
        pass

    @abstractmethod
    def close(self) -> None:
        pass

    def run(self) -> None:
        for entry in self.in_queue.listen():
            event = entry['data']
            if event.luid is None:
                break

            elif event.location:
                self.handle_event(event)

        self.close()