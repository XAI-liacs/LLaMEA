import numpy as np

class AdaptiveDifferentialEvolution:
    def __init__(self, budget=10000, dim=10, population_size=None, F=0.5, CR=0.9):
        self.budget = budget
        self.dim = dim
        self.population_size = population_size or 10 * dim
        self.F = F
        self.CR = CR
        self.f_opt = np.Inf
        self.x_opt = None
        self.success_rate = 0.0

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        population = np.random.uniform(lb, ub, (self.population_size, self.dim))
        fitness = np.array([func(ind) for ind in population])
        self.budget -= self.population_size
        best_idx = np.argmin(fitness)
        self.f_opt = fitness[best_idx]
        self.x_opt = population[best_idx]

        while self.budget > 0:
            successful_trials = 0
            for i in range(self.population_size):
                indices = np.arange(self.population_size)
                indices = np.delete(indices, i)
                a, b, c = population[np.random.choice(indices, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)

                crossover = np.random.rand(self.dim) < self.CR
                trial = np.where(crossover, mutant, population[i])
                
                f_trial = func(trial)
                self.budget -= 1
                if f_trial < fitness[i]:
                    fitness[i] = f_trial
                    population[i] = trial
                    successful_trials += 1
                    if f_trial < self.f_opt:
                        self.f_opt = f_trial
                        self.x_opt = trial

                if self.budget <= 0:
                    break
            
            self.update_parameters(successful_trials / self.population_size)

        return self.f_opt, self.x_opt

    def update_parameters(self, success_rate):
        if success_rate > 0.2:
            self.F = min(1.0, self.F + 0.1)
        else:
            self.F = max(0.1, self.F - 0.1)
        self.CR = 0.9 - 0.4 * success_rate  # dynamically adjust CR