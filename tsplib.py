"""Основной остров: отдельная популяция, которая эволюционирует в своём потоке.
Хранит данные в обычных (не потокобезопасных) структурах — им владеет ровно один поток."""
from typing import List

import numpy as np

from tspga.hop import Migrant
from tspga.instance import TSPInstance
from tspga.operators import Crossover, Mutation, Selection
from tspga.population import random_tours


class Island:
    def __init__(self, island_id: int, instance: TSPInstance, selection: Selection, crossover: Crossover,
                 mutation: Mutation, pop_size: int, elite_count: int, rng: np.random.Generator):
        if pop_size < 2 or pop_size % 2:
            raise ValueError("pop_size должен быть чётным числом не меньше 2")
        if not 0 <= elite_count < pop_size:
            raise ValueError("elite_count должен быть в диапазоне [0; pop_size)")
        self.island_id, self.instance, self.rng = island_id, instance, rng
        self.selection, self.crossover, self.mutation = selection, crossover, mutation
        self.pop_size, self.elite_count = pop_size, elite_count
        self.population = random_tours(rng, pop_size, instance.n_cities)
        self.costs = instance.tour_lengths(self.population)
        self.best_history: List[float] = []
        self.mean_history: List[float] = []
        self.snapshots: List[np.ndarray] = []              # лучший маршрут после каждой эпохи
        self._record()

    @property
    def best_cost(self) -> float:
        return float(self.costs.min())

    @property
    def best_tour(self) -> np.ndarray:
        return self.population[int(self.costs.argmin())].copy()

    def evolve(self, generations: int) -> None:
        for _ in range(generations):
            self._step()

    def snapshot(self) -> None:
        self.snapshots.append(self.best_tour)

    def emigrants(self, count: int) -> List[Migrant]:
        best = np.argsort(self.costs)[:count]
        return [Migrant(self.population[i].copy(), float(self.costs[i])) for i in best]

    def accept(self, migrants: List[Migrant]) -> None:
        """Мигрант заменяет худшую особь, если он лучше неё и такого маршрута в популяции ещё нет."""
        for m in sorted(migrants, key=lambda x: x.cost):
            worst = int(self.costs.argmax())
            if m.cost >= self.costs[worst] or (self.population == m.tour).all(axis=1).any():
                continue
            self.population[worst] = m.tour
            self.costs[worst] = m.cost

    def _step(self) -> None:
        parents = self.population[self.selection.select(-self.costs, self.pop_size, self.rng)]
        children = self.mutation.mutate(self.crossover.cross(parents, self.rng), self.rng)
        if self.elite_count:                                   # элитизм: лучшие переходят без изменений
            children[:self.elite_count] = self.population[np.argsort(self.costs)[:self.elite_count]]
        self.population = children
        self.costs = self.instance.tour_lengths(children)
        self._record()

    def _record(self) -> None:
        self.best_history.append(float(self.costs.min()))
        self.mean_history.append(float(self.costs.mean()))
