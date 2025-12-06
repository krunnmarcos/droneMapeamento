from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, Iterable, List

DEFAULT_COORDINATE_FILES = ("coordenadas.csv", "coordenadas(1).csv")


@dataclass(frozen=True)
class CoordinateRecord:
    cep: str
    longitude: float
    latitude: float

    def as_tuple(self) -> tuple[float, float]:
        return (self.latitude, self.longitude)


class CoordinatesLoader:
    def __init__(self, csv_path: str | Path | None = None) -> None:
        self.csv_path = self._resolve_path(csv_path)

    def load(self) -> List[CoordinateRecord]:
        if not self.csv_path.exists():
            raise FileNotFoundError(f"Coordinate file not found: {self.csv_path}")
        with self.csv_path.open("r", newline="", encoding="utf-8") as handle:
            reader = csv.DictReader(handle)
            self._validate_header(reader.fieldnames)
            records: List[CoordinateRecord] = []
            for row in reader:
                if not row or not row.get("cep"):
                    continue
                records.append(
                    CoordinateRecord(
                        cep=row["cep"].strip(),
                        longitude=float(row["longitude"]),
                        latitude=float(row["latitude"]),
                    )
                )
        return records

    def load_mapping(self) -> Dict[str, CoordinateRecord]:
        return {record.cep: record for record in self.load()}

    @staticmethod
    def _validate_header(fieldnames: Iterable[str] | None) -> None:
        expected = {"cep", "longitude", "latitude"}
        if not fieldnames or not expected.issubset({name.strip() for name in fieldnames}):
            raise ValueError("CSV header must contain: cep, longitude, latitude")

    @staticmethod
    def _resolve_path(csv_path: str | Path | None) -> Path:
        if csv_path is not None:
            return Path(csv_path)
        for candidate in DEFAULT_COORDINATE_FILES:
            candidate_path = Path(candidate)
            if candidate_path.exists():
                return candidate_path
        raise FileNotFoundError(
            "Coordinate file not provided and no default file was found (coordenadas.csv or coordenadas(1).csv)."
        )
