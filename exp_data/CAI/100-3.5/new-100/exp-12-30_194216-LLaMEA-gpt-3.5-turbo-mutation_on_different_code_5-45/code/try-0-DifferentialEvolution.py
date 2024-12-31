import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def opposite_direction_mutation(self, population, idx, F):
        r1, r2, r3 = np.random.choice(population, 3, replace=False)
        mutant = population[idx] + F * (r1 - r2) - F * (r3 - population[idx])
        return np.clip(mutant, -5.0, 5.0)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, size=(self.budget, self.dim))

        for i in range(self.budget):
            trial_population = np.copy(population)
            for j in range(self.dim):
                if np.random.rand() < self.CR or j == np.random.randint(0, self.dim):
                    trial_population[i, j] = self.opposite_direction_mutation(population, i, self.F)[j]

            f = func(population[i])
            f_trial = func(trial_population[i])

            if f_trial < f:
                population[i] = trial_population[i]
                f = f_trial

            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = population[i]

        return self.f_opt, self.x_opt