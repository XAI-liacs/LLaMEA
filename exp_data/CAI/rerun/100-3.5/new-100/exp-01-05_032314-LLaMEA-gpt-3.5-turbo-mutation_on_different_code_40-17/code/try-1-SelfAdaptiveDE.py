import numpy as np

class SelfAdaptiveDE:
    def __init__(self, budget=10000, dim=10, F_init=0.8, CR_init=0.9, F_lb=0.2, F_ub=0.9, CR_lb=0.1, CR_ub=0.9, strategy='best1bin'):
        self.budget = budget
        self.dim = dim
        self.F_init = F_init
        self.CR_init = CR_init
        self.F_lb = F_lb
        self.F_ub = F_ub
        self.CR_lb = CR_lb
        self.CR_ub = CR_ub
        self.strategy = strategy
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pop_size = 10 * self.dim
        bounds = (func.bounds.lb, func.bounds.ub)
        pop = np.random.uniform(bounds[0], bounds[1], (pop_size, self.dim))
        F_list = np.full(pop_size, self.F_init)
        CR_list = np.full(pop_size, self.CR_init)

        for i in range(self.budget):
            for j in range(pop_size):
                idxs = list(range(pop_size))
                idxs.remove(j)

                if self.strategy == 'best1bin':
                    a, b, c = np.random.choice(idxs, 3, replace=False)
                    x_best = pop[np.argmin([func(ind) for ind in pop])]
                    mutant = pop[a] + F_list[j] * (pop[b] - pop[c])
                    cross_points = np.random.rand(self.dim) < CR_list[j]
                    trial = np.where(cross_points, mutant, pop[j])
                else:
                    raise NotImplementedError("Strategy not implemented")

                f_trial = func(trial)
                if f_trial < func(pop[j]):
                    pop[j] = trial
                    F_list[j] = np.clip(F_list[j] + np.random.normal(0, 0.1), self.F_lb, self.F_ub)
                    CR_list[j] = np.clip(CR_list[j] + np.random.normal(0, 0.1), self.CR_lb, self.CR_ub)
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt