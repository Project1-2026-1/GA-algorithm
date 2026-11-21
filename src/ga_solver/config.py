from dataclasses import dataclass


@dataclass
class GAConfig:
    pop_size: int = 100
    generations: int = 200
    crossover_rate: float = 0.9
    mutation_rate: float = 0.2
    mutation_scale: float = 0.01
    tournament_k: int = 3
    elite_size: int = 2
    lower_bound: float = -10.0
    upper_bound: float = 10.0
    tolerance: float = 1e-8
    patience: int | None = 50
    seed: int | None = None

    def __post_init__(self) -> None:
        if self.lower_bound >= self.upper_bound:
            raise ValueError("lower_bound phải nhỏ hơn upper_bound")
        if self.pop_size < 2:
            raise ValueError("pop_size phải >= 2")
        if not 0 <= self.elite_size < self.pop_size:
            raise ValueError("elite_size phải nằm trong [0, pop_size)")
        if self.tournament_k < 1:
            raise ValueError("tournament_k phải >= 1")

    @property
    def mutation_std(self) -> float:
        return self.mutation_scale * (self.upper_bound - self.lower_bound)


@dataclass
class SolverConfig:
    n_subintervals: int = 20  // chia 20 khoảng 
    root_tolerance: float = 1e-6
    exclusion_radius: float = 1e-4
    min_width: float = 1e-3
    max_retries: int = 3
    sign_check_points: int = 50
    max_ga_runs: int = 2000

    def __post_init__(self) -> None:
        if self.n_subintervals < 1:
            raise ValueError("n_subintervals phải >= 1")
        if self.exclusion_radius <= 0:
            raise ValueError("exclusion_radius phải > 0")
        if self.min_width <= 0:
            raise ValueError("min_width phải > 0")
        if self.max_retries < 0:
            raise ValueError("max_retries phải >= 0")