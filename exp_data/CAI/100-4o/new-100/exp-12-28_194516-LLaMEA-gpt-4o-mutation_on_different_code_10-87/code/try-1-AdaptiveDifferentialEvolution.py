import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, pop_size=50):
        self.budget = budget
        self.dim = dim
        self.pop_size = pop_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.CR = 0.9  # Initial crossover probability
        self.F = 0.8   # Initial differential weight

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize population
        population = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        evaluations = self.pop_size

        while evaluations < self.budget:
            adaptive_pop_size = int(self.pop_size * (1 - evaluations / self.budget))
            for i in range(adaptive_pop_size):
                # Mutation: select three random individuals
                a, b, c = population[np.random.choice(self.pop_size, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                # Crossover
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, population[i])
                
                # Selection
                f_trial = func(trial)
                evaluations += 1
                if f_trial < fitness[i]:
                    population[i] = trial
                    fitness[i] = f_trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Self-adaptive parameters adjustment
                if np.random.rand() < 0.1:
                    self.CR = np.clip(self.CR + np.random.normal(0, 0.1), 0, 1)
                    self.F = np.clip(self.F + np.random.normal(0, 0.1), 0, 2)

                if evaluations >= self.budget:
                    break

        return self.f_opt, self.x_opt