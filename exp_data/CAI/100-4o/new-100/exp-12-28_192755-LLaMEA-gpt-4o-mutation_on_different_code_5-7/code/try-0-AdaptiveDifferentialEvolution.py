import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = max(5, dim * 10)
        self.budget_used = 0

    def __call__(self, func):
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget_used += self.population_size

        while self.budget_used < self.budget:
            F, CR = self.adaptive_parameters(fitness)
            for i in range(self.population_size):
                idxs = list(range(0, i)) + list(range(i + 1, self.population_size))
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c), func.bounds.lb, func.bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                f = func(trial)
                self.budget_used += 1
                if f < fitness[i]:
                    population[i] = trial
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
                if self.budget_used >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def adaptive_parameters(self, fitness):
        if len(fitness) < 2:
            return 0.5, 0.9
        improvement = np.diff(np.sort(fitness))
        avg_improvement = np.mean(improvement)
        F = np.clip(0.5 + (0.5 - 0.1) * (1 - avg_improvement), 0.1, 0.9)
        CR = np.clip(0.9 - (0.9 - 0.1) * avg_improvement, 0.1, 0.9)
        return F, CR