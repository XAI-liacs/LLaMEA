import numpy as np
from scipy.optimize import minimize

class AdaptiveDEPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 10 * dim
        self.mutation_factor = 0.5
        self.crossover_prob = 0.7
        self.population = None
        self.lb = None
        self.ub = None
        self.best = None
        self.best_score = np.inf

    def initialize_population(self, lb, ub):
        self.population = lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
        self.best = np.copy(self.population[0])

    def periodic_constraint(self, position):
        period = (self.ub - self.lb) / self.dim
        period_position = self.lb + (np.round((position - self.lb) / period) * period)
        return np.clip(period_position, self.lb, self.ub)

    def differential_evolution(self, func):
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                indices = np.random.choice(self.population_size, 3, replace=False)
                a, b, c = self.population[indices]
                mutant = self.periodic_constraint(a + self.mutation_factor * (b - c))
                crossover = np.random.rand(self.dim) < self.crossover_prob
                trial = np.where(crossover, mutant, self.population[i])
                trial = self.periodic_constraint(trial)

                trial_score = func(trial)
                if trial_score < func(self.population[i]):
                    self.population[i] = trial
                    if trial_score < self.best_score:
                        self.best = trial
                        self.best_score = trial_score

    def local_refinement(self, func):
        result = minimize(func, self.best, method='BFGS', 
                          bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.best = result.x
            self.best_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub)
        self.differential_evolution(func)
        self.local_refinement(func)
        return self.best