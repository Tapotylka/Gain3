"""Визуализация: маршруты на островах, графики лучшей и средней приспособленности, GIF-анимация."""
import math

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.animation import FuncAnimation, PillowWriter

BEST_COLOR = "#d62728"


def _grid(n, cols=2, cell=(4.6, 4.2)):
    rows = math.ceil(n / cols)
    fig, axes = plt.subplots(rows, cols, figsize=(cell[0] * cols, cell[1] * rows), squeeze=False)
    return fig, axes.ravel()


def _draw_route(ax, instance, tour, title, best=False):
    ax.clear()
    xy = instance.coords[np.append(tour, tour[0])]
    ax.plot(xy[:, 0], xy[:, 1], "-", lw=1.0, color=BEST_COLOR if best else "#1f77b4")
    ax.scatter(instance.coords[:, 0], instance.coords[:, 1], s=9, c="black", zorder=3)
    ax.set_title(title, fontsize=10, color=BEST_COLOR if best else "black", fontweight="bold" if best else "normal")
    ax.set_xticks([]), ax.set_yticks([])
    ax.set_aspect("equal")
    for spine in ax.spines.values():
        spine.set_edgecolor(BEST_COLOR if best else "gray")
        spine.set_linewidth(2.5 if best else 0.8)


def plot_routes(instance, result, path, epoch=None):
    """Маршрут каждого острова; лучшая популяция выделена красным.
    epoch=None — итоговые маршруты, иначе лучшие маршруты после эпохи epoch (нумерация с 0)."""
    n = len(result.best_tours)
    if epoch is None:
        tours = result.best_tours
    elif 0 <= epoch < len(result.snapshots[0]):
        tours = [result.snapshots[i][epoch] for i in range(n)]
    else:
        raise ValueError(f"epoch должен быть в диапазоне [0; {len(result.snapshots[0]) - 1}]")
    costs = [instance.tour_length(t) for t in tours]
    leader = int(np.argmin(costs))
    fig, axes = _grid(n)
    for i, tour in enumerate(tours):
        _draw_route(axes[i], instance, tour, f"{'★ ' if i == leader else ''}Остров {i + 1}: длина {costs[i]:.0f}"
                    f"{' (лучший)' if i == leader else ''}", i == leader)
    for ax in axes[n:]:
        ax.axis("off")
    fig.suptitle("Итоговые маршруты" if epoch is None else f"Маршруты после эпохи {epoch + 1}", fontsize=11)
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def plot_convergence(result, path):
    """Для каждого острова: лучшее (сплошная) и среднее (пунктир) значение длины маршрута по поколениям;
    последняя панель сравнивает лучшие значения всех островов."""
    n = len(result.best_history)
    fig, axes = _grid(n + 1, cell=(5.0, 3.4))
    for i in range(n):
        ax, best = axes[i], i == result.best_island
        g = np.arange(len(result.best_history[i]))
        ax.plot(g, result.best_history[i], color=BEST_COLOR if best else "#1f77b4", label="лучшее")
        ax.plot(g, result.mean_history[i], "--", color="gray", label="среднее")
        ax.set_title(f"{'★ ' if best else ''}Остров {i + 1}{' (лучшая популяция)' if best else ''}", fontsize=10,
                     color=BEST_COLOR if best else "black", fontweight="bold" if best else "normal")
        ax.set_xlabel("поколение"), ax.set_ylabel("длина маршрута (лог. шкала)")
        ax.set_yscale("log"), ax.grid(alpha=0.3, which="both"), ax.legend(fontsize=8)
        for spine in ax.spines.values():
            spine.set_edgecolor(BEST_COLOR if best else "gray")
            spine.set_linewidth(2.5 if best else 0.8)
    ax = axes[n]
    for i in range(n):
        best = i == result.best_island
        ax.plot(result.best_history[i], lw=2.5 if best else 1.0, color=BEST_COLOR if best else None,
                label=f"остров {i + 1}{' ★' if best else ''}", zorder=3 if best else 2)
    ax.set_title("Сравнение лучших значений", fontsize=10), ax.set_xlabel("поколение")
    ax.set_ylabel("длина маршрута (лог. шкала)"), ax.set_yscale("log")
    ax.grid(alpha=0.3, which="both"), ax.legend(fontsize=8)
    for extra in axes[n + 1:]:
        extra.axis("off")
    fig.tight_layout()
    fig.savefig(path, dpi=130)
    plt.close(fig)


def animate_routes(instance, result, path, fps=4, max_frames=60):
    """GIF: как меняются лучшие маршруты островов от эпохи к эпохе."""
    n = len(result.snapshots)
    epochs = len(result.snapshots[0])
    frames = np.unique(np.linspace(0, epochs - 1, min(max_frames, epochs)).astype(int))
    fig, axes = _grid(n, cell=(3.8, 3.5))

    def draw(k):
        e = frames[k]
        costs = [instance.tour_length(result.snapshots[i][e]) for i in range(n)]
        leader = int(np.argmin(costs))
        for i in range(n):
            _draw_route(axes[i], instance, result.snapshots[i][e],
                        f"{'★ ' if i == leader else ''}Остров {i + 1}: {costs[i]:.0f}", i == leader)
        fig.suptitle(f"Эпоха {e + 1} из {epochs}", fontsize=11)

    for ax in axes[n:]:
        ax.axis("off")
    FuncAnimation(fig, draw, frames=len(frames)).save(path, writer=PillowWriter(fps=fps))
    plt.close(fig)
