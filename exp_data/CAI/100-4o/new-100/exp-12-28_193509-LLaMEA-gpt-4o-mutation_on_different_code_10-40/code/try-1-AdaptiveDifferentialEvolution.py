import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        bounds_lb, bounds_ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(bounds_lb, bounds_ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])

        evals = self.pop_size
        while evals < self.budget:
            elite_idx = np.argmin(fitness)
            for i in range(self.pop_size):
                idxs = np.random.choice(np.delete(np.arange(self.pop_size), i), 3, replace=False)
                a, b, c = pop[idxs]
                mutant = np.clip(a + self.F * (b - c), bounds_lb, bounds_ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])
                
                f = func(trial)
                evals += 1
                if f < fitness[i]:
                    fitness[i] = f
                    pop[i] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial

                if evals >= self.budget:
                    break

            # Elite preservation
            pop[0] = np.copy(pop[elite_idx])
            fitness[0] = fitness[elite_idx]

            # Adaptive control of F and CR
            self.F = np.clip(0.5 + 0.1 * np.random.randn(), 0.4, 0.9)
            self.CR = np.clip(0.9 + 0.1 * np.random.randn(), 0.8, 1.0)

        return self.f_opt, self.x_opt