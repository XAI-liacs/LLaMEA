import numpy as np

class AdaptiveDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.NP = 10  # population size
        self.F = 0.5  # differential weight
        self.CR = 0.9  # crossover rate
        self.min_val = -5.0
        self.max_val = 5.0
        self.population = np.random.uniform(self.min_val, self.max_val, (self.NP, self.dim))
        self.best_solution = None

    def __call__(self, func):
        for _ in range(self.budget):
            new_population = []
            for i in range(self.NP):
                a, b, c = np.random.choice(self.NP, 3, replace=False)
                mutant = self.population[a] + self.F * (self.population[b] - self.population[c])
                mutant = np.clip(mutant, self.min_val, self.max_val)

                trial_vector = np.copy(self.population[i])
                for j in range(self.dim):
                    if np.random.rand() > self.CR:
                        trial_vector[j] = mutant[j]

                if func(trial_vector) < func(self.population[i]):
                    self.population[i] = trial_vector

                if self.best_solution is None or func(self.population[i]) < func(self.best_solution):
                    self.best_solution = np.copy(self.population[i])

            self.F = max(0.1, min(0.9, self.F + np.random.normal(0, 0.1)))
            self.CR = max(0.1, min(0.9, self.CR + np.random.normal(0, 0.1)))

        return self.best_solution