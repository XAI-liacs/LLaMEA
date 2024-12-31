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
        self.num_evals = 0
        
    def differential_evolution(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.num_evals += self.pop_size
        
        while self.num_evals < self.budget:
            for i in range(self.pop_size):
                indices = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                f = func(trial)
                self.num_evals += 1
                if f < fitness[i]:
                    fitness[i] = f
                    population[i] = trial

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = trial

                if self.num_evals >= self.budget:
                    break

            self.F = 0.5 + 0.3 * np.random.rand()  # Adapt F between generations
            self.CR = 0.8 + 0.2 * np.random.rand()  # Adapt CR between generations

        return self.f_opt, self.x_opt

    def adaptive_restart(self, func):
        while self.num_evals < self.budget:
            self.f_opt = np.Inf
            self.x_opt = None
            self.differential_evolution(func)

    def __call__(self, func):
        self.adaptive_restart(func)
        return self.f_opt, self.x_opt