import numpy as np

class OptimizedHybridDESA:
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
        self.beta_max = 0.85
        self.beta_min = 0.25
        self.alpha = 0.95  # Cooling parameter for simulated annealing
        self.current_budget = 0

    def __call__(self, func):
        while self.current_budget < self.budget:
            self.evaluate_population(func)
            self.perform_differential_evolution(func)
            self.apply_simulated_annealing(func)
        
        return self.best

    def evaluate_population(self, func):
        for i in np.where(self.fitness == np.inf)[0]:
            if self.current_budget >= self.budget:
                break
            self.fitness[i] = func(self.population[i])
            self.current_budget += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best = self.population[i].copy()

    def perform_differential_evolution(self, func):
        for i in range(self.population_size):
            if self.current_budget >= self.budget:
                break
            indices = np.delete(np.arange(self.population_size), i)
            a, b, c = np.random.choice(indices, 3, replace=False)
            beta = self.beta_min + (self.beta_max - self.beta_min) * (1 - self.current_budget / self.budget)
            mutant = np.clip(self.population[a] + beta * (self.population[b] - self.population[c]), self.lower_bound, self.upper_bound)
            trial = np.where(np.random.rand(self.dim) < 0.8, mutant, self.population[i])
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
            candidate = self.population[i] + np.random.normal(0, 0.05, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            self.current_budget += 1
            delta = candidate_fitness - self.fitness[i]
            if delta < 0 or np.random.rand() < np.exp(-delta / T):
                self.population[i] = candidate
                self.fitness[i] = candidate_fitness
                if candidate_fitness < self.best_fitness:
                    self.best_fitness = candidate_fitness
                    self.best = candidate
            T *= self.alpha