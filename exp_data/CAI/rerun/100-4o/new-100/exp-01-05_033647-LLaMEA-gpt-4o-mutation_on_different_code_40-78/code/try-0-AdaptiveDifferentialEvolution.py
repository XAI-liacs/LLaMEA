import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])

        for _ in range(self.budget - self.population_size):
            idxs = np.random.choice(self.population_size, 3, replace=False)
            a, b, c = population[idxs]
            mutant = np.clip(a + self.F * (b - c), lb, ub)

            j_rand = np.random.randint(self.dim)
            trial = np.array([mutant[j] if np.random.rand() < self.CR or j == j_rand else population[i][j] for j in range(self.dim)])
            
            f_trial = func(trial)
            if f_trial < fitness[i]:
                population[i] = trial
                fitness[i] = f_trial
                if f_trial < self.f_opt:
                    self.f_opt, self.x_opt = f_trial, trial

            # Adaptive parameters based on diversity
            diversity = np.mean(np.std(population, axis=0))
            self.F = 0.5 + diversity * 0.5
            self.CR = 0.1 + (1 - diversity) * 0.8

        return self.f_opt, self.x_opt