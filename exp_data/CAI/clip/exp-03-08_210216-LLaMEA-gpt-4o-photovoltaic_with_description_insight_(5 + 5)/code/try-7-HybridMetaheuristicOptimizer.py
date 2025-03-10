import numpy as np

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.current_budget = 0
        self.pop_size = min(50, dim * 5)  # Population size for DE
        self.population = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        self.fitness = np.full(self.pop_size, np.inf)
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7
        self.best_solution = None
        self.best_fitness = np.inf

    def differential_evolution(self, func):
        for i in range(self.pop_size):
            idxs = np.random.choice(self.pop_size, 3, replace=False)
            a, b, c = self.population[idxs]
            # Fix dimensionality handling in DE mutation
            mutant_vector = np.clip(a + self.mutation_factor * (b - c)[:self.dim], func.bounds.lb, func.bounds.ub)
            crossover = np.random.rand(self.dim) < self.crossover_rate
            trial_vector = np.where(crossover, mutant_vector, self.population[i])
            trial_fitness = self.evaluate(func, trial_vector)
            if trial_fitness < self.fitness[i]:
                self.population[i] = trial_vector
                self.fitness[i] = trial_fitness
                if trial_fitness < self.best_fitness:
                    self.best_fitness = trial_fitness
                    self.best_solution = trial_vector

    def local_search(self, func, solution):
        perturbation = np.random.normal(0, 0.1, self.dim)
        new_solution = np.clip(solution + perturbation, func.bounds.lb, func.bounds.ub)
        new_fitness = self.evaluate(func, new_solution)
        if new_fitness < self.best_fitness:
            self.best_solution = new_solution
            self.best_fitness = new_fitness

    def evaluate(self, func, solution):
        if self.current_budget < self.budget:
            self.current_budget += 1
            return func(solution)
        return np.inf

    def __call__(self, func):
        layers_step = max(2, self.dim // 10)
        for current_dim in range(layers_step, self.dim + 1, layers_step):
            self.population = np.random.uniform(-1, 1, (self.pop_size, current_dim))
            self.fitness = np.full(self.pop_size, np.inf)
            self.best_solution = None
            self.best_fitness = np.inf
            while self.current_budget < self.budget:
                self.differential_evolution(func)
                if self.best_solution is not None:
                    self.local_search(func, self.best_solution)
            if self.current_budget >= self.budget:
                break
        return self.best_solution