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
        self.population = np.random.uniform(-5.0, 5.0, (self.pop_size, self.dim))
        self.fitness = np.array([np.Inf]*self.pop_size)

    def __call__(self, func):
        evals = 0
        for i in range(self.pop_size):
            self.fitness[i] = func(self.population[i])
            evals += 1
            if self.fitness[i] < self.f_opt:
                self.f_opt = self.fitness[i]
                self.x_opt = self.population[i]

        while evals < self.budget:
            for i in range(self.pop_size):
                if evals >= self.budget:
                    break

                indices = [index for index in range(self.pop_size) if index != i]
                a, b, c = self.population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), -5.0, 5.0)

                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True

                trial = np.where(cross_points, mutant, self.population[i])
                f_trial = func(trial)
                evals += 1

                if f_trial < self.fitness[i]:
                    self.population[i] = trial
                    self.fitness[i] = f_trial

                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                # Adaptive control
                diversity = np.std(self.population, axis=0).mean()
                self.F = min(0.9, 0.5 + 0.5 * (diversity / (self.dim ** 0.5)))

        return self.f_opt, self.x_opt