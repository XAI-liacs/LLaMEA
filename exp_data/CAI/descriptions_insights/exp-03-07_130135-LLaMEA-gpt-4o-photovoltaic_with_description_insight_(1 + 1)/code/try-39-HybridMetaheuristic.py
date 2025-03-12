import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristic:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
        self.population_size = 50
        self.F = 0.8  # Differential weight
        self.CR = 0.9  # Crossover probability
        self.population = None
        self.best_solution = None
        self.best_score = float('-inf')

    def initialize_population(self, bounds):
        lower_bounds, upper_bounds = bounds.lb, bounds.ub
        self.population = np.random.uniform(lower_bounds, upper_bounds, (self.population_size, self.dim))
        
    def differential_evolution_step(self, bounds, func):
        lower_bounds, upper_bounds = bounds.lb, bounds.ub
        for i in range(self.population_size):
            idxs = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
            adaptive_F = np.random.rand() * self.F  # Adaptive mutation factor
            mutant = np.clip(a + adaptive_F * (b - c), lower_bounds, upper_bounds)
            cross_points = np.random.rand(self.dim) < (self.CR * (self.best_score + 1))
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, self.population[i])
            trial_score = func(trial)
            self.evals += 1
            if trial_score > self.best_score:
                self.best_score = trial_score
                self.best_solution = trial
            if trial_score > func(self.population[i]):
                self.population[i] = trial
                self.evals += 1
    
    def local_search(self, trial, bounds, func):
        result = minimize(func, trial, method='L-BFGS-B', bounds=list(zip(bounds.lb, bounds.ub)))
        self.evals += result.nfev
        return result.x, result.fun

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        while self.evals < self.budget:
            self.differential_evolution_step(bounds, func)
            if self.evals < self.budget:
                tournament_size = 2
                for i in range(self.population_size):
                    competitors = np.random.choice(self.population_size, tournament_size, replace=False)
                    winner = max(competitors, key=lambda x: func(self.population[x]))
                    self.population[i] = self.population[winner]
                    self.evals += tournament_size  # Adjust evals for additional function evaluations
        return self.best_solution