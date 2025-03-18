import numpy as np
from scipy.optimize import minimize

class HybridDEOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.F = 0.5
        self.CR = 0.9
        self.population = None
        self.best_solution = None
        self.best_score = np.inf
    
    def initialize_population(self, bounds):
        lb, ub = bounds.lb, bounds.ub
        self.population = np.random.uniform(lb, ub, (self.population_size, self.dim))
    
    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, ub
        noise_scale_base = 0.01
        for iteration in range(self.budget - self.population_size):
            noise_scale = noise_scale_base * (1 - iteration / self.budget)
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                F_dynamic = self.F * (1 - iteration / (3 * self.budget))
                CR_dynamic = self.CR * (1 - iteration / self.budget)
                mutant = np.clip(a + F_dynamic * (b - c), lb, ub)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                if not np.any(cross_points):
                    cross_points[np.random.randint(0, self.dim)] = True
                trial = np.where(cross_points, mutant, self.population[i]) + np.random.normal(0, noise_scale, self.dim)
                # Changed line:
                trial = (trial + np.roll(trial, shift=2)) / 2 + 0.1 * np.sin(np.linspace(0, np.pi * (1 + iteration / self.budget * 2), self.dim))  
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
            period_length = 4
            for i in range(0, self.dim, period_length):
                segment = self.best_solution[i:i + period_length]
                segment_median = np.median(segment)
                self.best_solution[i:i + period_length] = segment_median
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        self.differential_evolution(func, bounds)
        self.promote_periodicity()
        self.local_refinement(func, bounds)
        return self.best_solution