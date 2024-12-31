import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        # Adaptive parameters
        F_min, F_max = 0.4, 0.9
        CR_min, CR_max = 0.1, 0.9

        while evaluations < self.budget:
            for i in range(self.pop_size):
                # Mutation
                idxs = np.random.choice([j for j in range(self.pop_size) if j != i], 3, replace=False)
                x0, x1, x2 = population[idxs]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                # Update adaptive parameters
                self.F = F_min + (F_max - F_min) * np.random.rand()
                self.CR = CR_min + (CR_max - CR_min) * np.random.rand()

                if evaluations >= self.budget:
                    break

        best_idx = np.argmin(fitness)
        return fitness[best_idx], population[best_idx]