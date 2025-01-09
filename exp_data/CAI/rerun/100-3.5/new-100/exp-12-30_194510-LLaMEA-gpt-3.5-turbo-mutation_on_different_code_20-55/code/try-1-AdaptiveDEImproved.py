import numpy as np

class AdaptiveDEImproved:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 10
        self.cr = 0.9
        self.f_scale = 0.8

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        cr_adaptation = 0.1
        f_scale_adaptation = 0.1

        for _ in range(self.budget):
            trial_pop = np.zeros_like(pop)
            cr = np.clip(self.cr + cr_adaptation * np.random.randn(), 0, 1)
            f_scale = np.clip(self.f_scale + f_scale_adaptation * np.random.randn(), 0, 2)

            for i in range(self.pop_size):
                idxs = np.arange(self.pop_size)
                np.random.shuffle(idxs)
                a, b, c = pop[np.random.choice(idxs[:3], 3, replace=False)]
                mutant = a + f_scale * (b - c)
                cross_points = np.random.rand(self.dim) < cr
                trial_pop[i] = np.where(cross_points, mutant, pop[i])

            trial_fitness = np.array([func(ind) for ind in trial_pop])
            for i in range(self.pop_size):
                if trial_fitness[i] < fitness[i]:
                    pop[i] = trial_pop[i]
                    fitness[i] = trial_fitness[i]
                    if trial_fitness[i] < self.f_opt:
                        self.f_opt = trial_fitness[i]
                        self.x_opt = trial_pop[i]

        return self.f_opt, self.x_opt