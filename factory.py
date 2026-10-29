"""Островная модель с островами подскока: основные острова эволюционируют параллельно (в потоках),
мигранты передаются через потокобезопасные транзитные острова."""
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from typing import List, Optional

import numpy as np

from tspga.hop import HopIsland
from tspga.island import Island
from tspga.topology import Topology


@dataclass(frozen=True, eq=False)
class IslandModelResult:
    best_history: List[np.ndarray]      # по островам: лучшая длина маршрута в каждом поколении
    mean_history: List[np.ndarray]      # по островам: средняя длина маршрута в популяции
    snapshots: List[List[np.ndarray]]   # по островам: лучший маршрут после каждой эпохи
    best_costs: np.ndarray              # лучшая длина на каждом острове в конце
    best_tours: List[np.ndarray]
    best_island: int                    # «лучшая популяция»
    epochs_run: int
    evaluations: int
    elapsed: float                      # секунды
    migrants_sent: int
    migrants_received: int

    @property
    def best_cost(self) -> float:
        return float(self.best_costs[self.best_island])

    @property
    def best_tour(self) -> np.ndarray:
        return self.best_tours[self.best_island]


class IslandModel:
    def __init__(self, islands: List[Island], topology: Topology, epochs: int, interval: int, migrants: int,
                 hop_capacity: int = 16, patience_epochs: Optional[int] = None, synchronous: bool = True,
                 max_workers: Optional[int] = None):
        self.islands, self.epochs, self.interval, self.migrants = islands, epochs, interval, migrants
        self.patience_epochs, self.synchronous = patience_epochs, synchronous
        self.max_workers = max_workers or len(islands)
        self.hops = [HopIsland(s, t, hop_capacity) for s, t in topology.edges(len(islands))]
        self._incoming = [[h for h in self.hops if h.target == i] for i in range(len(islands))]
        self._outgoing = [[h for h in self.hops if h.source == i] for i in range(len(islands))]

    # --- одна эпоха острова = приём мигрантов -> interval поколений -> отправка лучших
    def _receive_and_evolve(self, i: int) -> None:
        island = self.islands[i]
        incoming = [m for hop in self._incoming[i] for m in hop.collect()]   # порядок рёбер фиксирован
        island.accept(incoming)
        island.evolve(self.interval)
        island.snapshot()

    def _send(self, i: int) -> None:
        if self._outgoing[i]:
            migrants = self.islands[i].emigrants(self.migrants)
            for hop in self._outgoing[i]:
                hop.deposit(migrants)

    def run(self) -> IslandModelResult:
        started = time.perf_counter()
        epochs_run = self._run_synchronous() if self.synchronous else self._run_asynchronous()
        elapsed = time.perf_counter() - started
        costs = np.array([isl.best_cost for isl in self.islands])
        return IslandModelResult(
            best_history=[np.array(i.best_history) for i in self.islands],
            mean_history=[np.array(i.mean_history) for i in self.islands],
            snapshots=[list(i.snapshots) for i in self.islands],
            best_costs=costs, best_tours=[i.best_tour for i in self.islands], best_island=int(costs.argmin()),
            epochs_run=epochs_run, elapsed=elapsed,
            evaluations=sum(len(i.best_history) * i.pop_size for i in self.islands),
            migrants_sent=sum(h.deposited for h in self.hops), migrants_received=sum(h.collected for h in self.hops),
        )

    def _run_synchronous(self) -> int:
        """Эпохи с барьером: результат не зависит от планирования потоков (воспроизводим при заданном seed)."""
        ids = range(len(self.islands))
        best_seen, stale, epochs_run = np.inf, 0, 0
        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            for _ in range(self.epochs):
                list(pool.map(self._receive_and_evolve, ids))
                list(pool.map(self._send, ids))
                epochs_run += 1
                current = min(i.best_cost for i in self.islands)
                best_seen, stale = (current, 0) if current < best_seen - 1e-12 else (best_seen, stale + 1)
                if self.patience_epochs is not None and stale >= self.patience_epochs:
                    break
        return epochs_run

    def _run_asynchronous(self) -> int:
        """Свободный режим: острова не ждут друг друга, обмен идёт только через транзитные острова."""
        def worker(i: int) -> None:
            for _ in range(self.epochs):
                self._receive_and_evolve(i)
                self._send(i)

        with ThreadPoolExecutor(max_workers=len(self.islands)) as pool:
            for future in [pool.submit(worker, i) for i in range(len(self.islands))]:
                future.result()                       # пробрасываем исключения из потоков
        return self.epochs
