"""Конфигурация и сборка островной модели."""
from dataclasses import dataclass
from typing import Optional

import numpy as np

from tspga.instance import TSPInstance
from tspga.island import Island
from tspga.model import IslandModel
from tspga.operators import InversionMutation, OrderCrossover, SwapMutation, TournamentSelection
from tspga.topology import FullyConnectedTopology, IsolatedTopology, RingTopology

TOPOLOGIES = {"ring": RingTopology, "full": FullyConnectedTopology, "none": IsolatedTopology}
MUTATIONS = {"inversion": InversionMutation, "swap": SwapMutation}


@dataclass(frozen=True)
class IslandConfig:
    n_islands: int = 4
    pop_size: int = 100              # размер популяции каждого острова
    epochs: int = 40                 # число эпох (между эпохами — обмен мигрантами)
    interval: int = 50               # поколений в эпохе
    migrants: int = 2                # сколько лучших особей отправляет остров за эпоху
    hop_capacity: int = 16           # вместимость транзитного острова
    topology: str = "ring"           # ring | full | none
    elite_count: int = 2
    tournament_size: int = 3
    crossover_rate: float = 0.9
    mutation: str = "inversion"      # inversion | swap
    mutation_rate: float = 0.3       # вероятность мутации особи
    seed: int = 0
    synchronous: bool = True         # True: воспроизводимые эпохи с барьером; False: свободные потоки
    patience_epochs: Optional[int] = None   # останов, если лучший маршрут не улучшался столько эпох
    max_workers: Optional[int] = None


def build_model(instance: TSPInstance, config: IslandConfig) -> IslandModel:
    c = config
    if c.n_islands < 1 or c.epochs < 1 or c.interval < 1:
        raise ValueError("n_islands, epochs и interval должны быть не меньше 1")
    if c.topology not in TOPOLOGIES:
        raise ValueError(f"неизвестная топология '{c.topology}', доступны: {sorted(TOPOLOGIES)}")
    if c.mutation not in MUTATIONS:
        raise ValueError(f"неизвестная мутация '{c.mutation}', доступны: {sorted(MUTATIONS)}")
    if not 0 <= c.migrants < c.pop_size:
        raise ValueError("migrants должен быть в диапазоне [0; pop_size)")
    seeds = np.random.SeedSequence(c.seed).spawn(c.n_islands)       # независимый генератор для каждого острова
    islands = [
        Island(i, instance, TournamentSelection(c.tournament_size), OrderCrossover(c.crossover_rate),
               MUTATIONS[c.mutation](c.mutation_rate), c.pop_size, c.elite_count, np.random.default_rng(seeds[i]))
        for i in range(c.n_islands)
    ]
    return IslandModel(islands, TOPOLOGIES[c.topology](), c.epochs, c.interval, c.migrants, c.hop_capacity,
                       c.patience_epochs, c.synchronous, c.max_workers)
