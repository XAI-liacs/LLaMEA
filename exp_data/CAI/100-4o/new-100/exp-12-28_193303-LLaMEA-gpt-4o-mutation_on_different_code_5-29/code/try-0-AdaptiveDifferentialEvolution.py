import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.f_opt = np.Inf
        self.x_opt = None
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.F = 0.5  # Initial mutation factor
        self.CR = 0.9  # Initial crossover probability

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])
                f_trial = func(trial)
                evals += 1

                if f_trial < func(self.population[i]):
                    self.population[i] = trial
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive adjustment of F and CR
                if evals % (self.budget // 10) == 0:  # Adjust every 10% of the budget
                    self.F = np.clip(self.F + np.random.normal(0, 0.1), 0.1, 1.0)
                    self.CR = np.clip(self.CR + np.random.normal(0, 0.1), 0.0, 1.0)

                if evals >= self.budget:
                    break

        return self.f_opt, self.x_opt