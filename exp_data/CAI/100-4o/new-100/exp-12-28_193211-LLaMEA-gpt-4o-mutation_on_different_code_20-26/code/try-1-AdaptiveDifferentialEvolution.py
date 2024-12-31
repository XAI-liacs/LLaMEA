import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=20, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lower_bound, upper_bound = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        self.update_best(fitness, pop)

        evals = self.pop_size
        diversity_threshold = 0.1
        while evals < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lower_bound, upper_bound)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                f_trial = func(trial)
                evals += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if evals >= self.budget:
                    break

            # Enhanced dynamic parameter adaptation and diversity check
            self.F = np.clip(np.mean(fitness) / (np.std(fitness) + 1e-8), 0.4, 0.9)
            self.CR = np.clip(0.5 + 0.3 * np.random.randn(), 0.2, 0.8)

            # Adjust population diversity
            pop_std = np.std(pop, axis=0)
            if np.mean(pop_std) < diversity_threshold:
                pop = np.random.uniform(lower_bound, upper_bound, (self.pop_size, self.dim))
                fitness = np.array([func(ind) for ind in pop])
                self.update_best(fitness, pop)
                evals += self.pop_size

        return self.f_opt, self.x_opt

    def update_best(self, fitness, pop):
        min_idx = np.argmin(fitness)
        if fitness[min_idx] < self.f_opt:
            self.f_opt = fitness[min_idx]
            self.x_opt = pop[min_idx]