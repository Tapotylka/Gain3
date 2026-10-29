"""Серии независимых запусков и статистика эффективности."""
from dataclasses import dataclass, replace
from typing import List, Optional

import numpy as np

from tspga.factory import IslandConfig, build_model
from tspga.instance import TSPInstance
from tspga.model import IslandModelResult


@dataclass(frozen=True, eq=False)
class SeriesResult:
    costs: np.ndarray
    times: np.ndarray
    results: List[IslandModelResult]


def run_series(instance: TSPInstance, config: IslandConfig, runs: int, base_seed: int = 0) -> SeriesResult:
    """runs независимых запусков с seed = base_seed, base_seed + 1, ..."""
    if runs < 1:
        raise ValueError("runs должен быть не меньше 1")
    results = [build_model(instance, replace(config, seed=base_seed + i)).run() for i in range(runs)]
    return SeriesResult(np.array([r.best_cost for r in results]), np.array([r.elapsed for r in results]), results)


def summarize(series: SeriesResult, optimum: Optional[float] = None) -> dict:
    c = series.costs
    out = {"runs": len(c), "mean": float(c.mean()), "std": float(c.std(ddof=1)) if len(c) > 1 else 0.0,
           "min": float(c.min()), "max": float(c.max()), "median": float(np.median(c)),
           "time_mean": float(series.times.mean())}
    if optimum:
        out["gap_mean_pct"] = (out["mean"] - optimum) / optimum * 100
        out["gap_best_pct"] = (out["min"] - optimum) / optimum * 100
    return out


def estimate_runs(pilot_costs, rel_error: float = 0.01, z: float = 1.96, minimum: int = 5, maximum: int = 100) -> int:
    """Число запусков, при котором 95 %-й доверительный интервал среднего уже ±rel_error от среднего:
    n = (z * s / (rel_error * mean))^2, s — выборочное СКО по пилотной серии."""
    pilot = np.asarray(pilot_costs, dtype=float)
    if len(pilot) < 2:
        raise ValueError("для оценки нужна пилотная серия минимум из двух запусков")
    n = int(np.ceil((z * pilot.std(ddof=1) / (rel_error * pilot.mean())) ** 2))
    return int(min(max(n, minimum), maximum))
