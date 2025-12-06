from __future__ import annotations

from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt

from solution_engine import LegDetail, TourStats


def plot_route(tour_stats: TourStats, filename: str | Path = "melhor_rota.png") -> Path:
    if not tour_stats.leg_details:
        raise ValueError("TourStats não contém leg_details para plotagem.")

    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    lats = [tour_stats.leg_details[0].start_coord[0]]
    lons = [tour_stats.leg_details[0].start_coord[1]]
    for leg in tour_stats.leg_details:
        lats.append(leg.end_coord[0])
        lons.append(leg.end_coord[1])

    fig, ax = plt.subplots(figsize=(8, 8))
    ax.plot(lons, lats, "-o", markersize=3, linewidth=1, alpha=0.8, label="Rota")
    ax.scatter(lons[0], lats[0], c="green", s=60, label="Base (início)")
    ax.scatter(lons[-1], lats[-1], c="red", s=60, label="Retorno")

    stops_lats = []
    stops_lons = []
    for leg in tour_stats.leg_details[:-1]:
        if leg.stop_required_at_destination:
            stops_lats.append(leg.end_coord[0])
            stops_lons.append(leg.end_coord[1])
    if stops_lats:
        ax.scatter(stops_lons, stops_lats, c="orange", s=25, label="Paradas/Recargas")

    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title("Rota do Drone (melhor solução)")
    ax.legend(loc="best")
    ax.grid(True, linewidth=0.3, alpha=0.5)
    fig.tight_layout()
    fig.savefig(output_path, dpi=150)
    plt.close(fig)
    return output_path
