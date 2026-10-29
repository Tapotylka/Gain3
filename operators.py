"""Топологии миграции: какие основные острова связаны транзитными островами (направленные рёбра)."""
from abc import ABC, abstractmethod
from typing import List, Tuple

Edge = Tuple[int, int]


class Topology(ABC):
    def edges(self, n_islands: int) -> List[Edge]:
        if n_islands < 1:
            raise ValueError("число островов должно быть не меньше 1")
        return self._edges(n_islands)

    @abstractmethod
    def _edges(self, n: int) -> List[Edge]:
        ...


class RingTopology(Topology):
    """i -> i+1: медленное распространение генов, высокое разнообразие."""

    def _edges(self, n):
        return [(i, (i + 1) % n) for i in range(n)] if n > 1 else []


class FullyConnectedTopology(Topology):
    """Каждый остров посылает мигрантов всем остальным: быстрая сходимость, меньше разнообразия."""

    def _edges(self, n):
        return [(i, j) for i in range(n) for j in range(n) if i != j]


class IsolatedTopology(Topology):
    """Без миграции: независимые популяции (контрольный вариант для сравнения)."""

    def _edges(self, n):
        return []
