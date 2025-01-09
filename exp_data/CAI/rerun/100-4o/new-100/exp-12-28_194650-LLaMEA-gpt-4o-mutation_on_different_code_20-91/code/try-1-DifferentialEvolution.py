import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None, F=0.5, CR=0.7):
        self.budget = budget
        self.dim = dim
        self.F = F
        self.CR = CR
        self.pop_size = pop_size if pop_size else dim * 10
        self.f_opt = np.Inf
        self.x_opt = None

    def adapt_F(self, iter_num, max_iter):
        return 0.4 + 0.6 * (1 - iter_num / max_iter)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        eval_count = self.pop_size
        max_iter = (self.budget - eval_count) // self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]

                F_dynamic = self.adapt_F(eval_count // self.pop_size, max_iter)
                mutant = np.clip(a + F_dynamic * (b - c), lb, ub)
                crossover_mask = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover_mask, mutant, population[i])

                f_trial = func(trial)
                eval_count += 1

                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial

                if f_trial < self.f_opt:
                    self.f_opt = f_trial
                    self.x_opt = trial

                if eval_count >= self.budget:
                    break

            best_idx = np.argmin(fitness)
            self.f_opt = fitness[best_idx]
            self.x_opt = population[best_idx]

        return self.f_opt, self.x_opt