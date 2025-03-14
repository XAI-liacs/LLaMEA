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
        self.stagnation_counter = 0  # Initialize stagnation counter

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
        CR_dynamic = self.CR_initial * (1 - self.current_evaluations / self.budget)
        fitness_variance = np.var(self.fitness)
        self.F = 0.5 + 0.2 * fitness_variance / (fitness_variance + 1)
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            dynamic_mutation = np.random.uniform(0.3, 0.8)  # Dynamic mutation strategy
            mutant = np.clip(a + dynamic_mutation * (b - c), func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < CR_dynamic
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_evaluations += 1

            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
                self.stagnation_counter = 0  # Reset stagnation counter on improvement
            else:
                self.stagnation_counter += 1
                if self.stagnation_counter > 5:  # Check for stagnation
                    self.initialize_population(func)
                    self.evaluate_population(func)

    def local_search(self, func):
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            candidate, candidate_fitness, _ = fmin_l_bfgs_b(func, self.population[i], bounds=list(zip(func.bounds.lb, func.bounds.ub)), approx_grad=True)
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