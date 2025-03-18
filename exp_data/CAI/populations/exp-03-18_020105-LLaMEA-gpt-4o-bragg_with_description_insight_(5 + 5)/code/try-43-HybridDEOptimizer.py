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
        sym_population = lb + ub - self.population[:self.population_size//2]
        self.population[:self.population_size//2] = (self.population[:self.population_size//2] + sym_population) / 2
    
    def differential_evolution(self, func, bounds):
        lb, ub = bounds.lb, bounds.ub
        for iter in range(self.budget - self.population_size):
            adaptive_F = self.F * (0.5 + 0.5 * (iter / (self.budget - self.population_size)))  # Adaptive mutation scaling
            for i in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != i]
                a, b, c = self.population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + adaptive_F * (b - c), lb, ub)
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
                if np.random.rand() < 0.1:  # Added periodic mutation operator
                    self.population[i] = self.periodic_mutation(self.population[i], lb, ub)

    def periodic_mutation(self, individual, lb, ub):  # New method for periodic mutation
        segment_size = self.dim // 5
        for i in range(0, self.dim, segment_size):
            mean_val = np.mean(individual[i:i+segment_size])
            individual[i:i+segment_size] = np.clip(mean_val + 0.2 * np.random.randn(segment_size), lb[i:i+segment_size], ub[i:i+segment_size])  # Increased from 0.1 to 0.2
        return individual
    
    def local_refinement(self, func, bounds):
        result = minimize(func, self.best_solution, bounds=[(bounds.lb[i], bounds.ub[i]) for i in range(self.dim)], method='L-BFGS-B')
        if result.fun < self.best_score:
            self.best_solution = result.x
            self.best_score = result.fun
    
    def promote_periodicity(self):
        if self.best_solution is not None:
            quarter_dim = self.dim // 4
            for i in range(0, self.dim, quarter_dim):
                segment = self.best_solution[i:i + quarter_dim]
                segment_mean = np.mean(segment)
                self.best_solution[i:i + quarter_dim] = segment_mean
    
    def __call__(self, func):
        bounds = func.bounds
        self.initialize_population(bounds)
        self.differential_evolution(func, bounds)
        self.promote_periodicity()
        self.local_refinement(func, bounds)
        return self.best_solution