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
        self.diversity_threshold = 1e-4  # Fitness diversity threshold

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
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
            F_dynamic = self.F * (1.0 + 0.5 * np.random.random())  # Dynamic mutation scaling
            mutant = np.clip(a + F_dynamic * (b - c), func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < CR_dynamic
            trial = np.where(crossover, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_evaluations += 1

            if trial_fitness < self.fitness[i] or np.std(self.fitness) < self.diversity_threshold:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

    def local_search(self, func):
        max_iter_limit = min(20, self.budget - self.current_evaluations)  # Dynamic iteration limit
        for i in range(self.population_size):
            if self.current_evaluations >= self.budget:
                break
            candidate, candidate_fitness, _ = fmin_l_bfgs_b(func, self.population[i], bounds=list(zip(func.bounds.lb, func.bounds.ub)), approx_grad=True, maxfun=max_iter_limit)
            self.current_evaluations += 1
            if candidate_fitness < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness

    def __call__(self, func):
        self.initialize_population(func)
        self.evaluate_population(func)
        
        # Adaptive dynamic population resizing based on evaluations
        if self.current_evaluations > self.budget * 0.5:
            self.population_size = int(self.population_size * 0.75)

        while self.current_evaluations < self.budget:
            self.differential_evolution_step(func)
            self.local_search(func)

        best_idx = np.argmin(self.fitness)
        return self.population[best_idx], self.fitness[best_idx]