import numpy as np
from scipy.optimize import fmin_l_bfgs_b

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR_initial = 0.9  # Initial crossover probability
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
        CR_dynamic = self.CR_initial * (1 - self.current_evaluations / self.budget)  # Adaptive CR
        fitness_variance = np.var(self.fitness)
        self.F = 0.5 + 0.2 * fitness_variance / (fitness_variance + 1)  # Adjust F based on fitness variance
        convergence_rate = np.mean(self.fitness) / np.min(self.fitness)
        if convergence_rate < 1.5:  # Dynamically adjust population size
            self.population_size = min(100 * self.dim, self.population_size + self.dim)
            self.initialize_population(func)
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < CR_dynamic
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_evaluations += 1

            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

    def local_search(self, func):
        leader_weight = 0.7
        leader_idx = np.argmin(self.fitness)
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            candidate_init = leader_weight * self.population[leader_idx] + (1 - leader_weight) * self.population[i]
            candidate, candidate_fitness, _ = fmin_l_bfgs_b(func, candidate_init, bounds=list(zip(func.bounds.lb, func.bounds.ub)), approx_grad=True)
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