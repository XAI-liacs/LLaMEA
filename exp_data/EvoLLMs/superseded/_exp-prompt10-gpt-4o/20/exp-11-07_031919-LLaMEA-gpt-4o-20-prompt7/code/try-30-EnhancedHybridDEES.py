import numpy as np
from multiprocessing import Pool

class EnhancedHybridDEES:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.fitness = np.full(self.population_size, np.inf)
        self.best = None
        self.best_fitness = np.inf
        self.beta_max = 0.9
        self.beta_min = 0.3
        self.mutation_prob = 0.8
        self.current_budget = 0

    def __call__(self, func):
        self.evaluate_population(func)
        while self.current_budget < self.budget:
            self.perform_differential_evolution(func)
            self.apply_simulated_annealing(func)
        return self.best

    def evaluate_population(self, func):
        with Pool() as pool:
            results = pool.map(func, self.population)
        for i, fitness in enumerate(results):
            if self.current_budget >= self.budget:
                break
            self.fitness[i] = fitness
            if fitness < self.best_fitness:
                self.best_fitness = fitness
                self.best = self.population[i].copy()
            self.current_budget += 1

    def perform_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            indices = np.delete(np.arange(self.population_size), i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            beta = self.beta_min + (self.beta_max - self.beta_min) * (self.best_fitness / (self.best_fitness + 1e-9))
            mutant = np.clip(self.population[a] + beta * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
            trial = np.where(np.random.rand(self.dim) < self.mutation_prob, mutant, self.population[i])
            trial_fitness = func(trial)
            self.current_budget += 1
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best = trial

    def apply_simulated_annealing(self, func):
        T = 1.0
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            candidate = self.population[i] + np.random.normal(0, 0.1 * T, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            self.current_budget += 1
            if candidate_fitness < self.fitness[i]:
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best = candidate