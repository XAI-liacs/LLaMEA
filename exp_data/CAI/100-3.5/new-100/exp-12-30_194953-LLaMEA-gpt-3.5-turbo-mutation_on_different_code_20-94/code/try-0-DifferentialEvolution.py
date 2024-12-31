import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def mutate(self, pop, target_idx):
        candidates = [idx for idx in range(len(pop)) if idx != target_idx]
        a, b, c = np.random.choice(candidates, 3, replace=False)
        return pop[a] + self.F * (pop[b] - pop[c])

    def crossover(self, target, trial_vector):
        mask = np.random.rand(self.dim) < self.CR
        offspring = np.where(mask, trial_vector, target)
        return offspring

    def select(self, func, target, trial):
        f_target = func(target)
        f_trial = func(trial)
        return trial if f_trial < f_target else target, f_trial

    def __call__(self, func):
        pop = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.budget, self.dim))

        for _ in range(self.budget):
            new_pop = np.empty_like(pop)
            for i in range(len(pop)):
                trial_vector = self.mutate(pop, i)
                trial = self.crossover(pop[i], trial_vector)
                new_pop[i], f_trial = self.select(func, pop[i], trial)

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = new_pop[i]

            pop = new_pop

        return self.f_opt, self.x_opt