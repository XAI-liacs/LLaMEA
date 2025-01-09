import numpy as np

class HybridDEARW:
    def __init__(self, budget, dim, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.best_solution = None
        self.best_fitness = float('inf')

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Differential Evolution mutation
                indices = np.random.choice(self.pop_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = np.clip(a + self.F * (b - c), self.lower_bound, self.upper_bound)

                # Crossover
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, self.population[i])

                # Adaptive Random Walk for exploration
                if np.random.rand() < 0.1:  # 10% chance to perform a random walk
                    step_size = np.random.rand() * (self.upper_bound - self.lower_bound) * 0.1
                    trial += np.random.uniform(-step_size, step_size, self.dim)
                    trial = np.clip(trial, self.lower_bound, self.upper_bound)

                # Selection
                trial_fitness = func(trial)
                evaluations += 1
                if trial_fitness < func(self.population[i]):
                    self.population[i] = trial
                    if trial_fitness < self.best_fitness:
                        self.best_solution = trial
                        self.best_fitness = trial_fitness
                
                if evaluations >= self.budget:
                    break

        return self.best_solution