import numpy as np

class EnhancedDE:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=20):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.F_lb = 0.2
        self.F_ub = 0.8
        self.CR_lb = 0.6
        self.CR_ub = 1.0

    def mutation(self, population, current_idx):
        idxs = [idx for idx in range(len(population)) if idx != current_idx]
        a, b, c = np.random.choice(idxs, 3, replace=False)
        self.F = np.clip(np.random.normal(self.F, 0.1), self.F_lb, self.F_ub)
        return population[a] + self.F * (population[b] - population[c])

    def crossover(self, target, mutant):
        crossover_prob = np.clip(np.random.normal(self.CR, 0.1), self.CR_lb, self.CR_ub)
        crossover_points = np.random.rand(self.dim) < crossover_prob
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