import numpy as np
from scipy.optimize import minimize

class HybridOptimizer:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.population_size = 20
        self.CR = 0.9
        self.F = 0.8

    def differential_evolution(self, func, bounds):
        population = np.random.uniform(bounds.lb, bounds.ub, (self.population_size, self.dim))
        for i in range(self.budget // 2):
            for j in range(self.population_size):
                idxs = [idx for idx in range(self.population_size) if idx != j]
                a, b, c = population[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + self.F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, population[j])
                f = func(trial)
                if f < func(population[j]):
                    population[j] = trial
                    if f < self.f_opt:
                        self.f_opt = f
                        self.x_opt = trial
        return self.x_opt

    def nelder_mead(self, func, x0):
        result = minimize(func, x0, method='Nelder-Mead', 
                          options={'maxfev': self.budget // 2, 'disp': False})
        if result.fun < self.f_opt:
            self.f_opt = result.fun
            self.x_opt = result.x

    def __call__(self, func):
        bounds = func.bounds
        best_de = self.differential_evolution(func, bounds)
        self.nelder_mead(func, best_de)
        return self.f_opt, self.x_opt