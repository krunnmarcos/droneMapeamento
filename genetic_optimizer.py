from __future__ import annotations

import random
from dataclasses import dataclass
from statistics import mean, pstdev
from typing import List, Sequence

from data_loader import CoordinateRecord, CoordinatesLoader
from flight_engine import BASE_CEP, WindForecast
from solution_engine import RouteEvaluator, TourStats
from solution_exporter import export_solution
from route_visualizer import plot_route


@dataclass
class IndividualRecord:
    genome: List[int]
    fitness: float
    stats: TourStats


class GeneticOptimizer:
    def __init__(
        self,
        coordinate_records: Sequence[CoordinateRecord],
        forecast: WindForecast,
        population_size: int = 120,
        mutation_rate: float = 0.04,
        tournament_size: int = 5,
        random_seed: int | None = None,
    ) -> None:
        if population_size < 2:
            raise ValueError("Population size must be at least 2.")
        if tournament_size < 2:
            raise ValueError("Tournament size must be at least 2.")
        self.population_size = population_size
        self.mutation_rate = mutation_rate
        self.tournament_size = tournament_size
        self.random = random.Random(random_seed)
        self.route_evaluator = RouteEvaluator(coordinate_records, forecast)
        self.target_ceps: List[str] = [record.cep for record in coordinate_records if record.cep != BASE_CEP]
        if not self.target_ceps:
            raise ValueError("Coordinate list must include at least one destination CEP besides the base.")
        self.genome_length = len(self.target_ceps)
        self.cep_to_index = {cep: idx for idx, cep in enumerate(self.target_ceps)}
        self.distance_cache: dict[tuple[str, str], float] = {}

    def run(self, generations: int = 50, log_interval: int = 10) -> IndividualRecord:
        population = self._initialize_population()
        best_overall: IndividualRecord | None = None
        for generation in range(1, generations + 1):
            evaluated = [self._evaluate_individual(genome) for genome in population]
            best_generation = max(evaluated, key=lambda record: record.fitness)
            if best_overall is None or best_generation.fitness > best_overall.fitness:
                best_overall = best_generation
            if log_interval and generation % log_interval == 0:
                self._print_generation_log(generation, evaluated, best_generation)
            population = self._produce_next_generation(population, evaluated)
        assert best_overall is not None
        self._print_generation_log(generations, [best_overall], best_overall, prefix="Final")
        return best_overall

    def decode_route(self, genome: Sequence[int]) -> List[str]:
        return [self.target_ceps[idx] for idx in genome]

    def _initialize_population(self) -> List[List[int]]:
        base_genome = list(range(self.genome_length))
        population: List[List[int]] = []
        population.append(self._generate_nearest_neighbor_genome())
        for _ in range(self.population_size - 1):
            genome = base_genome.copy()
            self.random.shuffle(genome)
            population.append(genome)
        return population

    def _generate_nearest_neighbor_genome(self) -> List[int]:
        remaining = set(self.target_ceps)
        current_cep = BASE_CEP
        ordered_ceps: List[str] = []
        coords = self.route_evaluator.coordinates
        physics = self.route_evaluator.physics
        while remaining:
            current_coord = coords[current_cep]
            next_cep = min(
                remaining,
                key=lambda candidate: physics.calculate_haversine(current_coord, coords[candidate]),
            )
            ordered_ceps.append(next_cep)
            remaining.remove(next_cep)
            current_cep = next_cep
        return [self.cep_to_index[cep] for cep in ordered_ceps]

    def _two_opt_improve(self, genome: List[int]) -> List[int]:
        best = genome.copy()
        improved = True
        while improved:
            improved = False
            for i in range(len(best) - 1):
                for j in range(i + 2, len(best) + 1):
                    if j - i == 1:
                        continue
                    delta = self._two_opt_delta(best, i, j)
                    if delta < -1e-6:
                        best[i:j] = reversed(best[i:j])
                        improved = True
                        break
                if improved:
                    break
        return best

    def _two_opt_delta(self, genome: Sequence[int], i: int, j: int) -> float:
        node_a = self._node_for_position(genome, i - 1)
        node_b = self._node_for_position(genome, i)
        node_c = self._node_for_position(genome, j - 1)
        node_d = self._node_for_position(genome, j)
        current = self._distance_between_nodes(node_a, node_b) + self._distance_between_nodes(node_c, node_d)
        proposed = self._distance_between_nodes(node_a, node_c) + self._distance_between_nodes(node_b, node_d)
        return proposed - current

    def _node_for_position(self, genome: Sequence[int], pos: int) -> str:
        if pos < 0 or pos >= len(genome):
            return BASE_CEP
        return self.target_ceps[genome[pos]]

    def _distance_between_nodes(self, cep_a: str, cep_b: str) -> float:
        if cep_a == cep_b:
            return 0.0
        key = (cep_a, cep_b) if cep_a < cep_b else (cep_b, cep_a)
        if key not in self.distance_cache:
            coord_a = self.route_evaluator.coordinates[cep_a]
            coord_b = self.route_evaluator.coordinates[cep_b]
            self.distance_cache[key] = self.route_evaluator.physics.calculate_haversine(coord_a, coord_b)
        return self.distance_cache[key]

    def _evaluate_individual(self, genome: Sequence[int]) -> IndividualRecord:
        cep_sequence = self.decode_route(genome)
        stats = self.route_evaluator.evaluate_route(cep_sequence)
        if not stats.valid:
            return IndividualRecord(list(genome), 1e-6, stats)
        total_minutes = stats.total_mission_duration_seconds / 60.0
        cost = total_minutes + (stats.total_stops * 180.0)
        if cost <= 0:
            cost = 1e-3
        fitness = 1.0 / cost
        return IndividualRecord(list(genome), fitness, stats)

    def _produce_next_generation(
        self,
        current_population: List[List[int]],
        evaluated: List[IndividualRecord],
    ) -> List[List[int]]:
        fitness_lookup = {id(genome): record.fitness for genome, record in zip(current_population, evaluated)}
        new_population: List[List[int]] = []
        elite = max(evaluated, key=lambda record: record.fitness)
        new_population.append(elite.genome.copy())
        while len(new_population) < self.population_size:
            parent_a = self._tournament_select(current_population, fitness_lookup)
            parent_b = self._tournament_select(current_population, fitness_lookup)
            child = self._order_crossover(parent_a, parent_b)
            self._mutate(child)
            new_population.append(child)
        return new_population

    def _tournament_select(self, population: List[List[int]], fitness_lookup: dict[int, float]) -> List[int]:
        competitors = self.random.sample(population, self.tournament_size)
        return max(competitors, key=lambda genome: fitness_lookup[id(genome)]).copy()

    def _order_crossover(self, parent_a: Sequence[int], parent_b: Sequence[int]) -> List[int]:
        length = len(parent_a)
        if length < 2:
            return list(parent_a)
        cut1, cut2 = sorted(self.random.sample(range(length), 2))
        child = [-1] * length
        child[cut1:cut2] = parent_a[cut1:cut2]
        pointer = cut2
        for gene in parent_b:
            if gene in child:
                continue
            if pointer >= length:
                pointer = 0
            child[pointer] = gene
            pointer += 1
        return child

    def _mutate(self, genome: List[int]) -> None:
        if self.genome_length < 2:
            return
        mutated = False
        if self.random.random() <= self.mutation_rate:
            i, j = self.random.sample(range(self.genome_length), 2)
            genome[i], genome[j] = genome[j], genome[i]
            mutated = True
        if mutated and self.genome_length >= 4 and self.random.random() < 0.1:
            self._two_opt_improve(genome)

    @staticmethod
    def _print_generation_log(
        generation: int,
        evaluated: List[IndividualRecord],
        best: IndividualRecord,
        prefix: str | None = None,
    ) -> None:
        label = prefix or f"Geração {generation}"
        fitness_values = [rec.fitness for rec in evaluated]
        avg = mean(fitness_values)
        std = pstdev(fitness_values) if len(fitness_values) > 1 else 0.0
        minutes = best.stats.total_mission_duration_seconds / 60.0
        print(
            f"{label}: BestFit={best.fitness:.6f} AvgFit={avg:.6f} StdFit={std:.6f} "
            f"TempoBest={minutes:.2f} min ParadasBest={best.stats.total_stops}"
        )


if __name__ == "__main__":
    loader = CoordinatesLoader()
    coordinates = loader.load()
    forecast = WindForecast.get_curitiba_forecast()
    optimizer = GeneticOptimizer(
        coordinate_records=coordinates,
        forecast=forecast,
        population_size=80,
        mutation_rate=0.03,
        tournament_size=3,
        random_seed=42,
    )
    best = optimizer.run(generations=50, log_interval=10)
    best_route = optimizer.decode_route(best.genome)
    print("=== MELHOR ROTA ENCONTRADA ===")
    print(best_route)
    export_path = export_solution(best.stats, "melhor_solucao.csv")
    print(f"CSV salvo em {export_path}")
    try:
        img_path = plot_route(best.stats, "melhor_rota.png")
        print(f"Gráfico salvo em {img_path}")
    except Exception as exc:  # pragma: no cover - visual aid only
        print(f"Falha ao gerar gráfico: {exc}")