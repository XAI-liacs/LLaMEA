import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9, pop_size=50, F_lower=0.1, F_upper=0.9, CR_lower=0.1, CR_upper=1.0):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size
        self.F_lower = F_lower
        self.F_upper = F_upper
        self.CR_lower = CR_lower
        self.CR_upper = CR_upper
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def rand_1_bin_mutate(pop, target_idx):
            a, b, c = np.random.choice(len(pop), 3, replace=False)
            mutant_vector = pop[a] + self.F * (pop[b] - pop[c])
            crossover_mask = np.random.rand(self.dim) < self.CR
            trial_vector = np.where(crossover_mask, mutant_vector, pop[target_idx])
            return trial_vector

        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))

        for i in range(self.budget):
            F = np.random.uniform(self.F_lower, self.F_upper)
            CR = np.random.uniform(self.CR_lower, self.CR_upper)
            for j in range(self.pop_size):
                trial_vector = rand_1_bin_mutate(pop, j)
                f = func(trial_vector)
                if f < func(pop[j]):
                    pop[j] = trial_vector
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_vector

        return self.f_opt, self.x_opt