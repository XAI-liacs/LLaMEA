import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        population = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))

        for i in range(self.budget):
            for j in range(pop_size):
                indices = [idx for idx in range(pop_size) if idx != j]
                a, b, c = np.random.choice(indices, 3, replace=False)

                mutant = population[a] + self.F * (population[b] - population[c])
                crossover = np.random.rand(self.dim) < self.CR
                trial_vector = np.where(crossover, mutant, population[j])

                f_trial = func(trial_vector)
                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial_vector

        return self.f_opt, self.x_opt