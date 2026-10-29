"""Начальная популяция: случайные перестановки городов."""
import numpy as np


def random_tours(rng: np.random.Generator, size: int, n_cities: int) -> np.ndarray:
    return rng.permuted(np.tile(np.arange(n_cities), (size, 1)), axis=1)
