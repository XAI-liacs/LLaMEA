import numpy as np

class HybridDEALS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 10 * dim
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.fitness = None

    def initialize_population(self):
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, 
                                            (self.population_size, self.dim))
        self.fitness = np.full(self.population_size, np.inf)

    def differential_evolution(self, func):
        for i in range(self.population_size):
            indices = np.random.choice(range(self.population_size), 3, replace=False)
            x0, x1, x2 = self.population[indices]
            mutant = np.clip(x0 + self.F * (x1 - x2), self.lower_bound, self.upper_bound)
            cross_points = np.random.rand(self.dim) < self.CR
            trial = np.where(cross_points, mutant, self.population[i])
            trial_fitness = func(trial)

            if trial_fitness < self.fitness[i]:
                self.population[i] = trial
                self.fitness[i] = trial_fitness

    def local_search(self, func, best_idx):
        step_size = 0.1
        for _ in range(5):  # Perform a fixed number of local steps
            candidate = self.population[best_idx] + np.random.uniform(-step_size, step_size, self.dim)
            candidate = np.clip(candidate, self.lower_bound, self.upper_bound)
            candidate_fitness = func(candidate)
            if candidate_fitness < self.fitness[best_idx]:
                self.population[best_idx] = candidate
                self.fitness[best_idx] = candidate_fitness

    def __call__(self, func):
        self.initialize_population()
        evaluations = 0
        while evaluations < self.budget:
            self.differential_evolution(func)
            evaluations += self.population_size
            best_idx = np.argmin(self.fitness)
            self.local_search(func, best_idx)
            evaluations += 5  # Local search evaluations
            if evaluations >= self.budget:
                break
        best_solution_idx = np.argmin(self.fitness)
        return self.population[best_solution_idx], self.fitness[best_solution_idx]