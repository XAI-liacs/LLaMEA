import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.best_fitness = np.Inf
        self.best_solution = None
        self.bounds = (-5.0, 5.0)
        self.F = 0.5
        self.CR = 0.9
        self.pop_size = min(10 * dim, budget // dim)
        self.population = np.random.uniform(self.bounds[0], self.bounds[1], (self.pop_size, dim))
        self.fitness = np.full(self.pop_size, np.Inf)

    def __call__(self, func):
        evaluations = 0

        # Evaluate initial population
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evaluations += 1
            if self.fitness[i] < self.best_fitness:
                self.best_fitness = self.fitness[i]
                self.best_solution = self.population[i].copy()

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Dynamically adjust F
                self.F = 0.4 + 0.1 * (self.budget - evaluations) / self.budget
                
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.bounds[0], self.bounds[1])

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = trial_fitness
                    if trial_fitness < self.best_fitness:
                        self.best_fitness = trial_fitness
                        self.best_solution = trial

                if evaluations >= self.budget:
                    break

        return self.best_fitness, self.best_solution