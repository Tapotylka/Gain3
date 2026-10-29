"""Задача коммивояжёра: города, матрица расстояний, длина маршрута."""
from dataclasses import dataclass
from typing import Optional

import numpy as np


def is_valid_tour(tour: np.ndarray, n_cities: int) -> bool:
    """Маршрут — перестановка всех n городов."""
    return len(tour) == n_cities and np.array_equal(np.sort(tour), np.arange(n_cities))


@dataclass(frozen=True, eq=False)
class TSPInstance:
    name: str
    coords: np.ndarray                  # (n, 2)
    dist: np.ndarray                    # (n, n)
    known_optimum: Optional[float] = None

    @classmethod
    def from_coords(cls, name, coords, rounded: bool = False, known_optimum: Optional[float] = None):
        """rounded=True — округление расстояний до целых, как EUC_2D в TSPLIB."""
        coords = np.asarray(coords, dtype=float)
        if coords.ndim != 2 or coords.shape[1] != 2 or len(coords) < 3:
            raise ValueError("нужно не менее трёх городов с двумя координатами")
        diff = coords[:, None, :] - coords[None, :, :]
        dist = np.sqrt((diff ** 2).sum(axis=2))
        if rounded:
            dist = np.floor(dist + 0.5)
        return cls(name, coords, dist, known_optimum)

    @property
    def n_cities(self) -> int:
        return len(self.coords)

    def tour_lengths(self, tours: np.ndarray) -> np.ndarray:
        """Длины замкнутых маршрутов для массива (m, n) — векторизованно."""
        return self.dist[tours, np.roll(tours, -1, axis=1)].sum(axis=1)

    def tour_length(self, tour: np.ndarray) -> float:
        return float(self.tour_lengths(np.asarray(tour)[None, :])[0])


def random_instance(n: int, seed: int = 0, size: float = 100.0) -> TSPInstance:
    coords = np.random.default_rng(seed).uniform(0, size, size=(n, 2))
    return TSPInstance.from_coords(f"random{n}", coords)


def circle_instance(n: int, radius: float = 100.0) -> TSPInstance:
    """Города на окружности: оптимальный маршрут — обход по кругу, его длина известна."""
    angles = 2 * np.pi * np.arange(n) / n
    coords = np.column_stack([radius * np.cos(angles), radius * np.sin(angles)])
    return TSPInstance.from_coords(f"circle{n}", coords, known_optimum=2 * n * radius * np.sin(np.pi / n))
