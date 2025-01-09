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
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.budget, self.dim))

        for i in range(self.budget):
            F_i = np.random.uniform(0.1, 0.9)  # Dynamic adaptation of F
            CR_i = np.random.uniform(0.1, 1.0)  # Dynamic adaptation of CR

            for j in range(self.budget):
                if j != i:
                    idxs = np.random.choice(self.dim, np.random.randint(1, self.dim), replace=False)
                    trial_vector = np.copy(population[i])
                    for idx in idxs:
                        trial_vector[idx] = population[j][idx]

                    f_trial = func(trial_vector)
                    if f_trial < func(population[i]):
                        population[i] = trial_vector

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial_vector

        return self.f_opt, self.x_opt