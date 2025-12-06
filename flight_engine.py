from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import date, datetime, time, timedelta
from typing import Iterable, List, Sequence, Tuple

EARTH_RADIUS_KM = 6371.0
MIN_SPEED_KMH = 36
MAX_SPEED_KMH = 96
SPEED_STEP_KMH = 4
OPERATION_PENALTY_SECONDS = 72
DAY_START = time(6, 0)
DAY_END = time(19, 0)
BASE_CEP = "82821020"
BASE_COORD = (-25.4233146347775, -49.2160678044742)


@dataclass(frozen=True)
class WindEntry:
    start: datetime
    end: datetime
    speed_kmh: float
    direction_from_deg: float  # Meteorological convention (coming from)


CARDINAL_TO_DEGREES = {
    "N": 0.0,
    "NNE": 22.5,
    "NE": 45.0,
    "ENE": 67.5,
    "E": 90.0,
    "ESE": 112.5,
    "SE": 135.0,
    "SSE": 157.5,
    "S": 180.0,
    "SSW": 202.5,
    "SW": 225.0,
    "WSW": 247.5,
    "W": 270.0,
    "WNW": 292.5,
    "NW": 315.0,
    "NNW": 337.5,
}

CURITIBA_FORECAST_TEMPLATE = [
    [
        (6, 17, "ENE"),
        (9, 18, "E"),
        (12, 19, "E"),
        (15, 19, "E"),
        (18, 20, "E"),
        (21, 20, "E"),
    ],
    [
        (6, 20, "E"),
        (9, 19, "E"),
        (12, 16, "E"),
        (15, 19, "E"),
        (18, 21, "E"),
        (21, 21, "E"),
    ],
    [
        (6, 15, "ENE"),
        (9, 17, "NE"),
        (12, 8, "NE"),
        (15, 20, "E"),
        (18, 16, "E"),
        (21, 15, "ENE"),
    ],
    [
        (6, 8, "ENE"),
        (9, 11, "ENE"),
        (12, 7, "NE"),
        (15, 6, "NE"),
        (18, 11, "E"),
        (21, 11, "E"),
    ],
    [
        (6, 3, "ENE"),
        (9, 3, "ENE"),
        (12, 7, "NE"),
        (15, 7, "NE"),
        (18, 10, "E"),
        (21, 11, "E"),
    ],
    [
        (6, 4, "NE"),
        (9, 5, "ENE"),
        (12, 4, "NE"),
        (15, 8, "E"),
        (18, 15, "E"),
        (21, 15, "E"),
    ],
    [
        (6, 6, "NE"),
        (9, 8, "NE"),
        (12, 14, "NE"),
        (15, 16, "NE"),
        (18, 13, "ENE"),
        (21, 10, "ENE"),
    ],
]


class WindForecast:
    """Wind forecast supporting multi-day schedules with periodic repetition."""

    def __init__(self, entries: Sequence[WindEntry], period_days: int | None = None):
        if not entries:
            raise ValueError("wind forecast requires at least one entry")
        self._entries: List[WindEntry] = sorted(entries, key=lambda e: e.start)
        self._base_datetime = self._entries[0].start
        self._period_days = period_days or max(1, (self._entries[-1].end.date() - self._base_datetime.date()).days + 1)
        self._period_seconds = self._period_days * 24 * 3600

    @classmethod
    def demo_day_one(cls, base_date: date | None = None) -> "WindForecast":
        """Creates a demo forecast for testing with 3-hour slices."""

        base_date = base_date or date(2025, 1, 1)
        pattern: Iterable[Tuple[int, float, float]] = (
            (0, 10.0, 135.0),
            (3, 12.0, 160.0),
            (6, 8.0, 180.0),
            (9, 14.0, 220.0),
            (12, 11.0, 250.0),
            (15, 7.0, 200.0),
            (18, 5.0, 170.0),
            (21, 9.0, 140.0),
        )
        entries: List[WindEntry] = []
        for start_hour, speed, direction in pattern:
            start_dt = datetime.combine(base_date, time(start_hour, 0))
            end_dt = start_dt + timedelta(hours=3)
            entries.append(WindEntry(start=start_dt, end=end_dt, speed_kmh=speed, direction_from_deg=direction))
        return cls(entries, period_days=1)

    @classmethod
    def get_curitiba_forecast(cls, base_date: date | None = None) -> "WindForecast":
        """Builds the 7-day forecast for Curitiba using the official table."""

        base_date = base_date or date(2025, 1, 1)
        entries: List[WindEntry] = []
        for day_offset, blocks in enumerate(CURITIBA_FORECAST_TEMPLATE):
            day = base_date + timedelta(days=day_offset)
            for start_hour, speed_kmh, direction_card in blocks:
                start_dt = datetime.combine(day, time(start_hour, 0))
                end_dt = start_dt + timedelta(hours=3)
                entries.append(
                    WindEntry(
                        start=start_dt,
                        end=end_dt,
                        speed_kmh=float(speed_kmh),
                        direction_from_deg=CARDINAL_TO_DEGREES[direction_card],
                    )
                )
        return cls(entries, period_days=len(CURITIBA_FORECAST_TEMPLATE))

    def get_wind_at(self, moment: datetime) -> WindEntry:
        """Returns the wind entry matching the provided moment, repeating every period."""

        base = self._base_datetime
        delta_seconds = (moment - base).total_seconds()
        normalized_seconds = delta_seconds % self._period_seconds
        normalized_moment = base + timedelta(seconds=normalized_seconds)
        for entry in self._entries:
            if entry.start <= normalized_moment < entry.end:
                return entry
        return self._entries[-1]


@dataclass(frozen=True)
class BatteryReport:
    autonomy_seconds: float
    consumed_seconds: float
    remaining_seconds: float
    operation_penalty_seconds: float
    requires_recharge: bool


class DronePhysics:
    """Encapsulates physical calculations for the Surveyor drone."""

    def calculate_haversine(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, coord1)
        lat2, lon2 = map(math.radians, coord2)
        delta_lat = lat2 - lat1
        delta_lon = lon2 - lon1
        a = math.sin(delta_lat / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin(delta_lon / 2) ** 2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))
        return EARTH_RADIUS_KM * c

    def calculate_heading(self, coord1: Tuple[float, float], coord2: Tuple[float, float]) -> float:
        lat1, lon1 = map(math.radians, coord1)
        lat2, lon2 = map(math.radians, coord2)
        delta_lon = lon2 - lon1
        x = math.sin(delta_lon) * math.cos(lat2)
        y = math.cos(lat1) * math.sin(lat2) - math.sin(lat1) * math.cos(lat2) * math.cos(delta_lon)
        bearing = math.degrees(math.atan2(x, y))
        return (bearing + 360) % 360

    def calculate_effective_velocity(
        self,
        drone_airspeed_kmh: float,
        drone_heading_deg: float,
        wind_speed_kmh: float,
        wind_direction_degrees: float,
    ) -> float:
        wind_heading_deg = (wind_direction_degrees + 180) % 360  # Convert from "coming from" to "going to"
        vx_drone, vy_drone = self._bearing_to_vector(drone_heading_deg, drone_airspeed_kmh)
        vx_wind, vy_wind = self._bearing_to_vector(wind_heading_deg, wind_speed_kmh)
        ground_vx = vx_drone + vx_wind
        ground_vy = vy_drone + vy_wind
        ground_speed = math.hypot(ground_vx, ground_vy)
        return ground_speed

    def calculate_battery_consumption(
        self,
        speed_kmh: float,
        flight_time_seconds: float,
        operation_events: int = 0,
    ) -> BatteryReport:
        autonomy_seconds = 0.93 * 5000 * (36 / speed_kmh) ** 2
        penalty_seconds = operation_events * OPERATION_PENALTY_SECONDS
        consumed = flight_time_seconds + penalty_seconds
        remaining = autonomy_seconds - consumed
        return BatteryReport(
            autonomy_seconds=autonomy_seconds,
            consumed_seconds=consumed,
            remaining_seconds=max(0.0, remaining),
            operation_penalty_seconds=penalty_seconds,
            requires_recharge=remaining < 0,
        )

    @staticmethod
    def _bearing_to_vector(bearing_deg: float, magnitude: float) -> Tuple[float, float]:
        radians = math.radians(bearing_deg)
        vx = math.sin(radians) * magnitude
        vy = math.cos(radians) * magnitude
        return vx, vy


@dataclass(frozen=True)
class FlightLegResult:
    distance_km: float
    effective_speed_kmh: float
    flight_time_seconds: float
    battery_consumed_seconds: float
    arrival_time: datetime
    additional_cost_reais: float
    needs_stop: bool


class FlightLegCalculator:
    def __init__(self, physics: DronePhysics, forecast: WindForecast) -> None:
        self.physics = physics
        self.forecast = forecast

    def simulate_leg(
        self,
        start_coord: Tuple[float, float],
        end_coord: Tuple[float, float],
        start_time: datetime,
        drone_speed_kmh: float,
    ) -> FlightLegResult:
        self._validate_speed(drone_speed_kmh)
        aligned_start, waited = self._align_with_window(start_time)
        wind_entry = self.forecast.get_wind_at(aligned_start)
        heading = self.physics.calculate_heading(start_coord, end_coord)
        effective_speed = self.physics.calculate_effective_velocity(
            drone_speed_kmh,
            heading,
            wind_entry.speed_kmh,
            wind_entry.direction_from_deg,
        )
        if effective_speed <= 0:
            raise ValueError("Ground speed must be positive; adjust heading or wind data.")
        distance = self.physics.calculate_haversine(start_coord, end_coord)
        flight_time_seconds = (distance / effective_speed) * 3600
        arrival_time, curfew_stop = self._apply_operational_window(aligned_start, flight_time_seconds)
        battery_report = self.physics.calculate_battery_consumption(
            speed_kmh=drone_speed_kmh,
            flight_time_seconds=flight_time_seconds,
            operation_events=3,  # takeoff, photo, landing
        )
        cost = 80.0 if arrival_time.time() > time(17, 0) else 0.0
        needs_stop = waited or curfew_stop or battery_report.requires_recharge
        return FlightLegResult(
            distance_km=distance,
            effective_speed_kmh=effective_speed,
            flight_time_seconds=flight_time_seconds,
            battery_consumed_seconds=battery_report.consumed_seconds,
            arrival_time=arrival_time,
            additional_cost_reais=cost,
            needs_stop=needs_stop,
        )

    def _align_with_window(self, start_time: datetime) -> Tuple[datetime, bool]:
        needs_wait = False
        if start_time.time() < DAY_START:
            start_time = datetime.combine(start_time.date(), DAY_START)
            needs_wait = True
        elif start_time.time() >= DAY_END:
            next_day = start_time.date() + timedelta(days=1)
            start_time = datetime.combine(next_day, DAY_START)
            needs_wait = True
        return start_time, needs_wait

    def _apply_operational_window(self, start_time: datetime, flight_seconds: float) -> Tuple[datetime, bool]:
        current = start_time
        remaining = flight_seconds
        needs_stop = False
        while remaining > 0:
            window_end = datetime.combine(current.date(), DAY_END)
            available = (window_end - current).total_seconds()
            if available <= 0:
                current = datetime.combine(current.date() + timedelta(days=1), DAY_START)
                needs_stop = True
                continue
            if remaining <= available:
                current += timedelta(seconds=remaining)
                remaining = 0
            else:
                current = window_end
                remaining -= available
                current = datetime.combine(current.date() + timedelta(days=1), DAY_START)
                needs_stop = True
        return current, needs_stop

    @staticmethod
    def _validate_speed(speed_kmh: float) -> None:
        if speed_kmh < MIN_SPEED_KMH or speed_kmh > MAX_SPEED_KMH:
            raise ValueError("Speed must stay within certified limits (36-96 km/h).")
        if (speed_kmh - MIN_SPEED_KMH) % SPEED_STEP_KMH != 0:
            raise ValueError("Speed adjustments must use 4 km/h increments.")


if __name__ == "__main__":
    physics = DronePhysics()
    wind_forecast = WindForecast.get_curitiba_forecast()
    point_a = (-25.4233146, -49.2160678)
    point_b = (-25.4936598, -49.3400481)
    drone_speed = 36.0
    drone_heading = physics.calculate_heading(point_a, point_b)
    distance_km = physics.calculate_haversine(point_a, point_b)
    effective_speed = physics.calculate_effective_velocity(
        drone_airspeed_kmh=drone_speed,
        drone_heading_deg=drone_heading,
        wind_speed_kmh=9.0,
        wind_direction_degrees=157.5,
    )
    flight_time_seconds = (distance_km / effective_speed) * 3600
    battery_report = physics.calculate_battery_consumption(
        speed_kmh=drone_speed,
        flight_time_seconds=flight_time_seconds,
        operation_events=3,
    )

    print("=== Teste de Validação (Ponto A -> Ponto B) ===")
    print(f"Distância (km): {distance_km:.3f}")
    print(f"Velocidade Efetiva (km/h): {effective_speed:.3f}")
    print(f"Tempo de Voo (s): {flight_time_seconds:.1f}")
    print(f"Consumo de Bateria (s): {battery_report.consumed_seconds:.1f}")
