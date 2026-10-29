"""Остров подскока (транзитный остров): потокобезопасный буфер, через который мигранты
«перепрыгивают» с одного основного острова на другой."""
import queue
import threading
from dataclasses import dataclass
from typing import List

import numpy as np


@dataclass(frozen=True, eq=False)
class Migrant:
    tour: np.ndarray
    cost: float


class HopIsland:
    """Основные острова работают с обычными массивами, а обмен идёт только через эту коллекцию.

    Внутри — queue.Queue (потокобезопасная FIFO-очередь). При переполнении вытесняются самые старые
    мигранты, поэтому ни отправитель, ни получатель никогда не блокируются.
    """

    def __init__(self, source: int, target: int, capacity: int = 16):
        if capacity < 1:
            raise ValueError("capacity должен быть не меньше 1")
        self.source, self.target, self.capacity = source, target, capacity
        self._queue: "queue.Queue[Migrant]" = queue.Queue()
        self._lock = threading.Lock()           # защищает пары «положить + вытеснить» и счётчики
        self._deposited = 0
        self._collected = 0

    @property
    def deposited(self) -> int:
        with self._lock:
            return self._deposited

    @property
    def collected(self) -> int:
        with self._lock:
            return self._collected

    def deposit(self, migrants: List[Migrant]) -> None:
        with self._lock:
            for m in migrants:
                self._queue.put_nowait(m)
                self._deposited += 1
                while self._queue.qsize() > self.capacity:
                    self._queue.get_nowait()

    def collect(self) -> List[Migrant]:
        """Забирает всех накопленных мигрантов (в порядке поступления)."""
        taken: List[Migrant] = []
        while True:
            try:
                taken.append(self._queue.get_nowait())
            except queue.Empty:
                break
        with self._lock:
            self._collected += len(taken)
        return taken
