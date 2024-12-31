import numpy as np

class AdaptiveLevyDE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10 * dim  # Population size
        self.F = 0.5  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.alpha = 1.5  # Levy exponent
        self.sigma = (np.gamma(1 + self.alpha) * np.sin(np.pi * self.alpha / 2) /
                      (np.gamma((1 + self.alpha) / 2) * self.alpha * 2 ** ((self.alpha - 1) / 2))) ** (1 / self.alpha)
        self.success_rate = 0.2  # Start with a moderate success rate

    def levy_flight(self, size):
        u = np.random.normal(0, self.sigma, size)
        v = np.random.normal(0, 1, size)
        step = u / (np.abs(v) ** (1 / self.alpha))
        return step

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.apply_along_axis(func, 1, population)

        for _ in range(self.budget // self.pop_size):
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                trial_fitness = func(trial)

                if trial_fitness < fitness[i]:
                    population[i] = trial
                    fitness[i] = trial_fitness
                    self.success_rate = 0.9 * self.success_rate + 0.1
                    if trial_fitness < self.f_opt:
                        self.f_opt = trial_fitness
                        self.x_opt = trial
                else:
                    self.success_rate = 0.9 * self.success_rate

                if np.random.rand() < self.success_rate:
                    population[i] = np.clip(population[i] + self.levy_flight(self.dim), lb, ub)

        return self.f_opt, self.x_opt