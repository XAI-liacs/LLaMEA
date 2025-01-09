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
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(self.lb, self.ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        budget_used = self.pop_size

        # Find initial best
        best_idx = np.argmin(fitness)
        self.f_opt, self.x_opt = fitness[best_idx], population[best_idx].copy()

        while budget_used < self.budget:
            trial_population = []

            for i in range(self.pop_size):
                idxs = [idx for idx in range(self.pop_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, population[i])
                trial_population.append(trial)

            new_fitness = []
            for i in range(self.pop_size):
                f = func(trial_population[i])
                budget_used += 1
                if f < fitness[i]:
                    population[i] = trial_population[i]
                    fitness[i] = f
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial_population[i]
                new_fitness.append(f)

                if budget_used >= self.budget:
                    break

            # Local Search Phase (optional)
            for i in range(self.pop_size):
                if np.random.rand() < 0.1:  # 10% probability
                    local_search = population[i] + np.random.uniform(-0.1, 0.1, self.dim)
                    local_search = np.clip(local_search, self.lb, self.ub)
                    f_local = func(local_search)
                    budget_used += 1
                    if f_local < fitness[i]:
                        population[i] = local_search
                        fitness[i] = f_local
                        if f_local < self.f_opt:
                            self.f_opt = f_local
                            self.x_opt = local_search
                    if budget_used >= self.budget:
                        break

        return self.f_opt, self.x_opt