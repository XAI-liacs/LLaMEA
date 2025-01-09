import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None

    def mutation(self, population, current_idx):
        idxs = [idx for idx in range(len(population)) if idx != current_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, mutant):
        crossover_points = np.random.rand(self.dim) < self.CR
        trial = np.where(crossover_points, mutant, target)
        return np.clip(trial, -5.0, 5.0)

    def __call__(self, func):
        population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        for _ in range(self.budget):
            for i in range(self.pop_size):
                x_target = population[i]
                x_mutant = self.mutation(population, i)
                x_trial = self.crossover(x_target, x_mutant)

                f_target = func(x_target)
                f_trial = func(x_trial)

                if f_trial < f_target:
                    population[i] = x_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = x_trial

        return self.f_opt, self.x_opt