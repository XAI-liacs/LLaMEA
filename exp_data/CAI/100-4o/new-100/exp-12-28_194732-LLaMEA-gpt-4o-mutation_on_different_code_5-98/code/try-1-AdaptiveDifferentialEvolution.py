import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 10 * dim
        self.F = 0.5  # initial mutation factor
        self.CR = 0.9 # initial crossover probability
        self.avg_improvement = 0.0  # Track average improvement

    def __call__(self, func):
        # Initialize population
        population = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.population_size, self.dim))
        fitness = np.array([func(individual) for individual in population])
        self.update_optimal(population, fitness)
        budget_used = self.population_size

        while budget_used < self.budget:
            for i in range(self.population_size):
                # Mutation
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), func.bounds.lb, func.bounds.ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])

                # Selection
                f_trial = func(trial)
                budget_used += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptation of F and CR
                self.adapt_parameters(f_trial, fitness[i], i)

                if budget_used >= self.budget:
                    break

        return self.f_opt, self.x_opt

    def update_optimal(self, population, fitness):
        idx_min = np.argmin(fitness)
        if fitness[idx_min] < self.f_opt:
            self.f_opt = fitness[idx_min]
            self.x_opt = population[idx_min]

    def adapt_parameters(self, f_trial, f_current, i):
        improvement = f_current - f_trial
        self.avg_improvement = 0.9 * self.avg_improvement + 0.1 * improvement
        if f_trial < f_current and improvement > self.avg_improvement:
            self.F = min(1, self.F + 0.1)
            self.CR = max(0.1, self.CR - 0.1)
        else:
            self.F = max(0.1, self.F - 0.1)
            self.CR = min(1, self.CR + 0.1)