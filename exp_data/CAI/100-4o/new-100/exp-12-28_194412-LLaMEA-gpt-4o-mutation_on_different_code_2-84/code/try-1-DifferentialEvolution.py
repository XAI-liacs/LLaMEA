import numpy as np

class DifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=None, F=0.8, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size or 10 * dim
        self.F = F  # Scale factor for mutation
        self.CR = CR  # Crossover probability
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in pop])
        budget_used = self.pop_size

        while budget_used < self.budget:
            for i in range(self.pop_size):
                if budget_used >= self.budget:
                    break

                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = pop[np.random.choice(indices, 3, replace=False)]

                # Adaptive F based on fitness improvement
                self.F = 0.5 + 0.5 * (1 - self.f_opt / (np.min(fitness) + 1e-8))
                
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, pop[i])

                f_trial = func(trial)
                budget_used += 1

                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    pop[i] = trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

        return self.f_opt, self.x_opt