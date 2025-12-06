from __future__ import annotations

import math
import unittest
from datetime import datetime

from flight_engine import BASE_COORD, DAY_START, DronePhysics, FlightLegCalculator, WindForecast


class TestDronePhysics(unittest.TestCase):
    def setUp(self) -> None:
        self.physics = DronePhysics()

    def test_effective_velocity_matches_pdf_reference(self) -> None:
        start = (-25.4233146, -49.2160678)
        destination = (-25.4133146, -49.3160678)
        heading = self.physics.calculate_heading(start, destination)
        effective_velocity = self.physics.calculate_effective_velocity(
            drone_airspeed_kmh=36.0,
            drone_heading_deg=heading,
            wind_speed_kmh=9.0,
            wind_direction_degrees=157.5,
        )
        self.assertAlmostEqual(effective_velocity, 41.0, places=0)

    def test_battery_autonomy_respects_correction_factor(self) -> None:
        report = self.physics.calculate_battery_consumption(speed_kmh=36.0, flight_time_seconds=0.0)
        self.assertAlmostEqual(report.autonomy_seconds, 4650.0, places=1)

    def test_operation_penalty_added_to_consumption(self) -> None:
        report = self.physics.calculate_battery_consumption(
            speed_kmh=36.0,
            flight_time_seconds=120.0,
            operation_events=3,
        )
        expected_consumption = 120.0 + (3 * 72.0)
        self.assertEqual(report.consumed_seconds, expected_consumption)


class TestFlightOperations(unittest.TestCase):
    def setUp(self) -> None:
        self.physics = DronePhysics()
        self.forecast = WindForecast.get_curitiba_forecast()
        self.leg_calculator = FlightLegCalculator(self.physics, self.forecast)

    def test_curfew_enforced_for_late_departure(self) -> None:
        start_time = datetime(2025, 1, 1, 18, 55)
        start_coord = BASE_COORD
        end_coord = (BASE_COORD[0] - 0.05, BASE_COORD[1])
        result = self.leg_calculator.simulate_leg(
            start_coord=start_coord,
            end_coord=end_coord,
            start_time=start_time,
            drone_speed_kmh=36.0,
        )
        self.assertTrue(result.needs_stop)
        self.assertGreaterEqual(result.arrival_time.time(), DAY_START)


if __name__ == "__main__":
    unittest.main()
