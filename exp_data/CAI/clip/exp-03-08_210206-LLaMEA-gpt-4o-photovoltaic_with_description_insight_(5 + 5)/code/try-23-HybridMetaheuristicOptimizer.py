import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.current_evaluations = 0

    def initialize_population(self, func):
        self.population = np.random.rand(self.population_size, self.dim)
        self.population = self.population * (func.bounds.ub - func.bounds.lb) + func.bounds.lb
        self.fitness = np.full(self.population_size, np.inf)

    def evaluate_population(self, func):
        for i in range(self.population_size):
            if self.fitness[i] == np.inf:
                self.fitness[i] = func(self.population[i])
                self.current_evaluations += 1

    def differential_evolution_step(self, func):
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            adaptive_F = self.F * (1 + (np.min(self.fitness) - self.fitness[i]) / (np.ptp(self.fitness) + 1e-9))
            mutant = np.clip(a + adaptive_F * (b - c), func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < self.CR
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_evaluations += 1

            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

    def local_search(self, func):
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            precision_factor = 0.05 * (1.0 / (1.0 + self.fitness[i]))
            perturb = np.random.normal(0, precision_factor, self.dim)
            candidate = np.clip(self.population[i] + perturb, func.bounds.lb, func.bounds.ub)
            candidate_fitness = func(candidate)
            self.current_evaluations += 1

            if candidate_fitness < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness

    def __call__(self, func):
        self.initialize_population(func)
        self.evaluate_population(func)

        while self.current_evaluations < self.budget:
            self.differential_evolution_step(func)
            self.local_search(func)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]