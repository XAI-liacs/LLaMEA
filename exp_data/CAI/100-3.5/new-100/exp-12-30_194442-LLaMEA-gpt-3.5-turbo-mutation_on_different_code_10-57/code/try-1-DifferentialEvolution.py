import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))
        F = 0.5 + 0.3 * np.random.rand()  # Adaptive parameter F
        CR = 0.1 + 0.8 * np.random.rand()  # Adaptive parameter CR

        for i in range(self.budget):
            for j in range(self.budget):
                idxs = [idx for idx in range(self.budget) if idx != i and idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]

                trial_vector = population[i] + F * (a - b)
                mask = np.random.rand(self.dim) < CR
                trial_vector = np.where(mask, trial_vector, population[i])

                f_trial = func(trial_vector)
                if f_trial < func(population[i]):
                    population[i] = trial_vector

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial_vector

        return self.f_opt, self.x_opt