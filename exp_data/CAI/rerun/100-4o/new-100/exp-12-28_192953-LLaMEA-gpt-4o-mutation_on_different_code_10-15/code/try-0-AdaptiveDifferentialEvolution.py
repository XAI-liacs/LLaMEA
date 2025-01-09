import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=50):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.apply_along_axis(func, 1, pop)
        self.f_opt = np.min(fitness)
        self.x_opt = pop[np.argmin(fitness)].copy()

        count_evals = self.population_size
        F, CR = 0.5, 0.9

        while count_evals < self.budget:
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                x1, x2, x3 = pop[indices]
                mutant = np.clip(x1 + F * (x2 - x3), lb, ub)
                cross_points = np.random.rand(self.dim) < CR
                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                count_evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial.copy()

                if count_evals >= self.budget:
                    break

            diversity = np.mean(np.std(pop, axis=0)) / (ub - lb)
            F = 0.4 + 0.2 * diversity
            CR = 0.8 + 0.1 * diversity

        return self.f_opt, self.x_opt