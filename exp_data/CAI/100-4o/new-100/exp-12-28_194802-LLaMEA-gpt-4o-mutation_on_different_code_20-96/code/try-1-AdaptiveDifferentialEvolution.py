import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.F_min, self.F_max = 0.5, 1.0
        self.CR_min, self.CR_max = 0.1, 0.9

    def __call__(self, func):
        # Initialize population
        bounds = func.bounds
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        
        evaluations = self.population_size

        while evaluations < self.budget:
            F = np.random.uniform(self.F_min, self.F_max)
            CR = np.random.uniform(self.CR_min, self.CR_max)

            for i in range(self.population_size):
                indices = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                mutant = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                crossover = np.random.rand(self.dim) < CR
                trial = np.where(crossover, mutant, population[i])

                f_trial = func(trial)
                evaluations += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

            idx_best = np.argmin(fitness)
            elite = population[idx_best]
            population[np.random.randint(self.population_size)] = elite

            self.population_size = max(10, int(self.population_size * 0.9)) if evaluations > self.budget * 0.5 else self.population_size

            if evaluations >= self.budget:
                break

        return self.f_opt, self.x_opt