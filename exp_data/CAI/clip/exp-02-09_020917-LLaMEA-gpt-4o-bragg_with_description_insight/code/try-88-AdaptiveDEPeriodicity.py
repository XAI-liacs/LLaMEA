import numpy as np
from scipy.optimize import minimize

class AdaptiveDEPeriodicity:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 8 * dim
        self.population = None
        self.lb = None
        self.ub = None
        self.F = 0.5  # Differential weight
        self.CR = 0.9 # Crossover probability
        self.gbest = None
        self.gbest_score = np.inf

    def initialize_population(self, lb, ub, size):
        self.population = lb + (ub - lb) * np.random.rand(size, self.dim)

    def periodic_weighting(self, iteration):
        return np.sin((iteration / self.budget) * np.pi) ** 2

    def differential_evolution(self, func):
        for iteration in range(self.budget - self.population_size):
            for i in range(self.population_size):
                candidates = list(range(self.population_size))
                candidates.remove(i)
                a, b, c = self.population[np.random.choice(candidates, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), self.lb, self.ub)
                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.population[i])
                trial_score = func(trial)
                
                if trial_score < func(self.population[i]):
                    self.population[i] = trial
                    if trial_score < self.gbest_score:
                        self.gbest = trial
                        self.gbest_score = trial_score
            
            # Apply periodic weighting
            self.F = self.periodic_weighting(iteration)
            self.CR = self.periodic_weighting(iteration)

    def local_refinement(self, func):
        result = minimize(func, self.gbest, method='BFGS', bounds=[(self.lb[i], self.ub[i]) for i in range(self.dim)])
        if result.success:
            self.gbest = result.x
            self.gbest_score = func(result.x)

    def __call__(self, func):
        self.lb, self.ub = func.bounds.lb, func.bounds.ub
        self.initialize_population(self.lb, self.ub, self.population_size)
        self.differential_evolution(func)
        self.local_refinement(func)
        return self.gbest