from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Sequence, Tuple

from data_loader import CoordinateRecord, CoordinatesLoader
from flight_engine import (
    BASE_COORD,
    BASE_CEP,
    DAY_END,
    DAY_START,
    MAX_SPEED_KMH,
    MIN_SPEED_KMH,
    OPERATION_PENALTY_SECONDS,
    SPEED_STEP_KMH,
    DronePhysics,
    FlightLegCalculator,
    WindForecast,
)

OPERATIONS_PER_LEG = 3
MISSION_START = datetime(2025, 1, 1, DAY_START.hour, DAY_START.minute, 0)
SPEED_CHOICES = list(range(MIN_SPEED_KMH, MAX_SPEED_KMH + 1, SPEED_STEP_KMH))


@dataclass
class SpeedOption:
    speed_kmh: int
    distance_km: float
    flight_time_seconds: float
    energy_required: float


@dataclass
class LegDetail:
    origin_cep: str
    destination_cep: str
    start_coord: Tuple[float, float]
    end_coord: Tuple[float, float]
    start_time: datetime
    end_time: datetime
    day_index: int
    speed_kmh: int
    distance_km: float
    flight_time_seconds: float
    stop_required_at_destination: bool = False


@dataclass
class TourStats:
    total_distance_km: float
    total_flight_time_seconds: float
    total_mission_duration_seconds: float
    total_stops: int
    valid: bool
    mission_start_time: datetime
    leg_details: List[LegDetail] = field(default_factory=list)
    log: List[str] = field(default_factory=list)


class RouteEvaluator:
    def __init__(
        self,
        coordinate_records: Sequence[CoordinateRecord],
        forecast: WindForecast,
        physics: DronePhysics | None = None,
    ) -> None:
        self.physics = physics or DronePhysics()
        self.forecast = forecast
        self.leg_calculator = FlightLegCalculator(self.physics, forecast)
        self.coordinates: Dict[str, tuple[float, float]] = {record.cep: record.as_tuple() for record in coordinate_records}
        self.coordinates.setdefault(BASE_CEP, BASE_COORD)
        self.speed_choices_asc = sorted(SPEED_CHOICES)
        self.energy_capacity = self._compute_energy_capacity()
        self.operations_penalty_seconds = OPERATIONS_PER_LEG * OPERATION_PENALTY_SECONDS

    def _compute_energy_capacity(self) -> float:
        baseline_autonomy = self.physics.calculate_battery_consumption(speed_kmh=MIN_SPEED_KMH, flight_time_seconds=0).autonomy_seconds
        return baseline_autonomy * (MIN_SPEED_KMH ** 2)

    @staticmethod
    def _seconds_available_from_energy(energy_units: float, reference_speed_kmh: float) -> float:
        if reference_speed_kmh <= 0:
            return 0.0
        return max(0.0, energy_units) / (reference_speed_kmh ** 2)

    def _energy_required(self, speed_kmh: int, flight_time_seconds: float) -> float:
        total_seconds = flight_time_seconds + self.operations_penalty_seconds
        return total_seconds * (speed_kmh ** 2)

    def _build_speed_options(
        self,
        start_coord: Tuple[float, float],
        end_coord: Tuple[float, float],
        start_time: datetime,
    ) -> List[SpeedOption]:
        options: List[SpeedOption] = []
        for speed in self.speed_choices_asc:
            try:
                distance_km, flight_time_seconds = self._estimate_leg(start_coord, end_coord, start_time, speed)
            except ValueError:
                continue
            energy_required = self._energy_required(speed, flight_time_seconds)
            options.append(
                SpeedOption(
                    speed_kmh=speed,
                    distance_km=distance_km,
                    flight_time_seconds=flight_time_seconds,
                    energy_required=energy_required,
                )
            )
        return options

    def evaluate_route(self, cep_sequence: Sequence[str]) -> TourStats:
        battery_energy = self.energy_capacity
        route = [BASE_CEP, *cep_sequence, BASE_CEP]
        current_time = MISSION_START
        mission_start = current_time
        total_distance = 0.0
        total_flight_time = 0.0
        total_stops = 0
        log: List[str] = [f"Início da missão às {current_time:%Y-%m-%d %H:%M}"]
        mission_failed = False
        leg_details: List[LegDetail] = []

        for origin, destination in zip(route[:-1], route[1:]):
            if mission_failed:
                break
            start_coord = self._get_coord(origin)
            end_coord = self._get_coord(destination)
            selected_option: SpeedOption | None = None
            leg_ready = False
            while not leg_ready and not mission_failed:
                current_time, wait_logged = self._ensure_within_window(current_time, log)
                if wait_logged:
                    total_stops += 1
                    battery_energy = self.energy_capacity
                    if leg_details:
                        leg_details[-1].stop_required_at_destination = True
                    continue
                speed_options = self._build_speed_options(start_coord, end_coord, current_time)
                if not speed_options:
                    mission_failed = True
                    log.append(f"Falha: Nenhuma velocidade válida para {origin}->{destination} (vento extremo).")
                    break
                window_end = datetime.combine(current_time.date(), DAY_END)
                options_before_curfew = [
                    option
                    for option in speed_options
                    if current_time + timedelta(seconds=option.flight_time_seconds) <= window_end
                ]
                if not options_before_curfew:
                    next_start = datetime.combine(current_time.date() + timedelta(days=1), DAY_START)
                    log.append(
                        f"{current_time:%Y-%m-%d %H:%M} - {origin}->{destination} adiado: chegaria após o toque de recolher. Pernoite até {next_start:%Y-%m-%d %H:%M}."
                    )
                    current_time = next_start
                    total_stops += 1
                    battery_energy = self.energy_capacity
                    if leg_details:
                        leg_details[-1].stop_required_at_destination = True
                    continue
                selected_option = None
                for option in options_before_curfew:
                    if option.energy_required <= battery_energy + 1e-6:
                        selected_option = option
                        break
                if selected_option is not None:
                    leg_ready = True
                    break
                min_energy = min(option.energy_required for option in options_before_curfew)
                if min_energy > self.energy_capacity + 1e-6:
                    mission_failed = True
                    log.append(
                        f"Falha: Mesmo com bateria cheia o trecho {origin}->{destination} excede a capacidade do drone."
                    )
                    break
                battery_energy = self.energy_capacity
                total_stops += 1
                if leg_details:
                    leg_details[-1].stop_required_at_destination = True
                log.append(
                    f"{current_time:%Y-%m-%d %H:%M} - Bateria insuficiente para {origin}->{destination}. Recarga realizada antes da decolagem."
                )
            if mission_failed or selected_option is None:
                break
            leg_start = current_time
            leg_result = self.leg_calculator.simulate_leg(
                start_coord=start_coord,
                end_coord=end_coord,
                start_time=leg_start,
                drone_speed_kmh=selected_option.speed_kmh,
            )
            total_distance += leg_result.distance_km
            total_flight_time += leg_result.flight_time_seconds
            battery_energy = max(0.0, battery_energy - selected_option.energy_required)
            battery_ref_seconds = self._seconds_available_from_energy(battery_energy, MIN_SPEED_KMH)
            log.append(
                (
                    f"Decolagem {origin}->{destination} às {leg_start:%Y-%m-%d %H:%M} a {selected_option.speed_kmh} km/h, "
                    f"chegada às {leg_result.arrival_time:%Y-%m-%d %H:%M}, bateria equivalente a {battery_ref_seconds:.1f}s @ {MIN_SPEED_KMH} km/h."
                )
            )
            day_index = (leg_start.date() - mission_start.date()).days + 1
            leg_details.append(
                LegDetail(
                    origin_cep=origin,
                    destination_cep=destination,
                    start_coord=start_coord,
                    end_coord=end_coord,
                    start_time=leg_start,
                    end_time=leg_result.arrival_time,
                    day_index=day_index,
                    speed_kmh=selected_option.speed_kmh,
                    distance_km=leg_result.distance_km,
                    flight_time_seconds=leg_result.flight_time_seconds,
                )
            )
            current_time = leg_result.arrival_time
            if battery_energy < -1e-6:
                mission_failed = True
                log.append("Falha: bateria esgotada em voo.")
                break

        total_duration = (current_time - mission_start).total_seconds()
        if leg_details:
            leg_details[-1].stop_required_at_destination = True
        return TourStats(
            total_distance_km=total_distance,
            total_flight_time_seconds=total_flight_time,
            total_mission_duration_seconds=total_duration,
            total_stops=total_stops,
            valid=not mission_failed,
            mission_start_time=mission_start,
            leg_details=leg_details,
            log=log,
        )

    def _get_coord(self, cep: str) -> tuple[float, float]:
        if cep not in self.coordinates:
            raise KeyError(f"CEP {cep} não encontrado nas coordenadas carregadas.")
        return self.coordinates[cep]

    def _ensure_within_window(self, current_time: datetime, log: List[str]) -> tuple[datetime, bool]:
        if current_time.time() < DAY_START:
            aligned = datetime.combine(current_time.date(), DAY_START)
            log.append(f"{current_time:%Y-%m-%d %H:%M} - Aguardando janela de voo até {aligned:%Y-%m-%d %H:%M}.")
            return aligned, True
        if current_time.time() >= DAY_END:
            next_start = datetime.combine(current_time.date() + timedelta(days=1), DAY_START)
            log.append(f"{current_time:%Y-%m-%d %H:%M} - Fora da janela. Retomando às {next_start:%Y-%m-%d %H:%M}.")
            return next_start, True
        return current_time, False

    def _estimate_leg(
        self,
        start_coord: tuple[float, float],
        end_coord: tuple[float, float],
        start_time: datetime,
        drone_speed_kmh: float,
    ) -> tuple[float, float]:
        heading = self.physics.calculate_heading(start_coord, end_coord)
        wind = self.forecast.get_wind_at(start_time)
        effective_speed = self.physics.calculate_effective_velocity(
            drone_airspeed_kmh=drone_speed_kmh,
            drone_heading_deg=heading,
            wind_speed_kmh=wind.speed_kmh,
            wind_direction_degrees=wind.direction_from_deg,
        )
        if effective_speed <= 0:
            raise ValueError("Velocidade efetiva inválida; ajuste a previsão de vento.")
        distance_km = self.physics.calculate_haversine(start_coord, end_coord)
        flight_time_seconds = (distance_km / effective_speed) * 3600
        return distance_km, flight_time_seconds


def _build_demo_csv(path: Path) -> None:
    rows = [
        ("cep", "longitude", "latitude"),
        ("82821020", -49.2160678, -25.4233146),
        ("81350686", -49.3400481, -25.4936598),
        ("82530380", -49.2336060, -25.4300626),
        ("82930390", -49.2047594, -25.4608672),
        ("81320083", -49.3183277, -25.4770955),
    ]
    with path.open("w", newline="", encoding="utf-8") as handle:
        for row in rows:
            handle.write(",".join(str(value) for value in row) + "\n")


if __name__ == "__main__":
    demo_path = Path("demo_coords.csv")
    _build_demo_csv(demo_path)
    loader = CoordinatesLoader(demo_path)
    coords = loader.load()
    forecast = WindForecast.get_curitiba_forecast()
    evaluator = RouteEvaluator(coords, forecast)
    route = ["81350686", "82530380"]
    stats = evaluator.evaluate_route(route)
    print("=== LOG DA ROTA ===")
    for entry in stats.log:
        print(entry)
    print("=== RESUMO ===")
    print(f"Distância total: {stats.total_distance_km:.2f} km")
    print(f"Tempo de voo: {stats.total_flight_time_seconds/60:.1f} min")
    print(f"Duração de missão: {stats.total_mission_duration_seconds/3600:.2f} h")
    print(f"Paradas (recargas/pernoites): {stats.total_stops}")
    print(f"Missão válida: {stats.valid}")
