import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = max(10, int(0.5 * dim))  # dynamic population size based on dimension
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_solution = None
        self.best_score = np.inf
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        for _ in range(self.budget - self.population_size):
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i])
                trial_score = func(trial)
                if trial_score < func(self.population[i]):
                    self.population[i] = trial
                    if trial_score < self.best_score:
                        self.best_score = trial_score
                        self.best_solution = trial
    
    def local_refinement(self, func, bounds):
        result = minimize(func, self.best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        if result.fun < self.best_score:
            self.best_solution = result.x
            self.best_score = result.fun
    
    def promote_periodicity(self):
        if self.best_solution is not None:
            half_dim = self.dim // 2
            for i in range(0, self.dim, half_dim):
                segment = self.best_solution[i:i + half_dim]
                segment_mean = np.mean(segment)
                self.best_solution[i:i + half_dim] = segment_mean
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        self.differential_evolution(func, bounds)
        self.promote_periodicity()
        self.local_refinement(func, bounds)
        return self.best_solution