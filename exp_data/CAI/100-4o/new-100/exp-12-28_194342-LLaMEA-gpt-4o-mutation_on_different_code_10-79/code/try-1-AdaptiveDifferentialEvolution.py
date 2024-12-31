import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size or 10 * dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        evaluations = self.pop_size
        F = 0.5
        CR = 0.9

        while evaluations < self.budget:
            for i in range(self.pop_size):
                if evaluations >= self.budget:
                    break

                # Mutation
                indices = np.random.choice(self.pop_size, 5, replace=False)  # Changed from 3 to 5
                while i in indices:
                    indices = np.random.choice(self.pop_size, 5, replace=False)  # Changed from 3 to 5
                a, b, c, d, e = population[indices]  # Added two more vectors d, e
                mutant = np.clip(a + F * (b - c + d - e), bounds[0], bounds[1])  # Expanded mutation with d, e

                # Crossover
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                trial_fitness = func(trial)
                evaluations += 1

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial

                # Adaptive parameters update
                F = 0.5 + 0.3 * np.sin(0.5 * evaluations / self.budget * np.pi)
                CR = 0.5 + 0.4 * np.cos(0.5 * evaluations / self.budget * np.pi)

        return self.f_opt, self.x_opt