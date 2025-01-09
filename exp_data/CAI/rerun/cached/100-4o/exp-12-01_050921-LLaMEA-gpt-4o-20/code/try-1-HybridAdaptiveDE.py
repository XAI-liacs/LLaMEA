import numpy as np

class HybridAdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20 + 5 * np.log(self.dim)
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (int(self.population_size), self.dim))
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability

    def __call__(self, func):
        evaluations = 0
        best_individual = None
        best_fitness = float('inf')

        while evaluations < self.budget:
            for i in range(int(self.population_size)):
                if evaluations >= self.budget:
                    break

                # Mutation
                idxs = [idx for idx in range(int(self.population_size)) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = a + self.F * (b - c)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])

                # Selection
                f_trial = func(trial)
                evaluations += 1

                if f_trial < func(self.population[i]):
                    self.population[i] = trial
                    if f_trial < best_fitness:
                        best_fitness = f_trial
                        best_individual = trial

                # Adaptive parameters
                if evaluations % (self.budget // 10) == 0:
                    self.F = np.random.uniform(0.4, 0.9)
                    self.CR = np.random.uniform(0.1, 1.0)

        return best_individual