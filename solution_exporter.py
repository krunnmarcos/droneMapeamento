from __future__ import annotations

import csv
from pathlib import Path

from solution_engine import LegDetail, TourStats


def export_solution(tour_stats: TourStats, filename: str | Path = "solucao.csv") -> Path:
    """Exporta o plano detalhado de voo para o formato solicitado pelo edital."""

    if not tour_stats.leg_details:
        raise ValueError("TourStats não contém detalhes de pernas para exportação.")
    output_path = Path(filename)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    mission_start_date = tour_stats.mission_start_time.date()
    header = [
        "CEP inicial",
        "Latitude inicial",
        "Longitude inicial",
        "Dia do voo",
        "Hora inicial",
        "Velocidade (km/h)",
        "CEP final",
        "Latitude final",
        "Longitude final",
        "Pouso",
        "Hora final",
    ]
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        for detail in tour_stats.leg_details:
            day_index = (detail.start_time.date() - mission_start_date).days + 1
            writer.writerow(
                [
                    detail.origin_cep,
                    f"{detail.start_coord[0]:.7f}",
                    f"{detail.start_coord[1]:.7f}",
                    day_index,
                    detail.start_time.strftime("%H:%M:%S"),
                    detail.speed_kmh,
                    detail.destination_cep,
                    f"{detail.end_coord[0]:.7f}",
                    f"{detail.end_coord[1]:.7f}",
                    "SIM" if detail.stop_required_at_destination else "NAO",
                    detail.end_time.strftime("%H:%M:%S"),
                ]
            )
    return output_path
