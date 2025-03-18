import numpy as np
from scipy.optimize import minimize

class HybridMetaheuristicOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def quasi_oppositional_initialization(self, bounds):
        mid_point = (bounds.ub + bounds.lb) / 2
        range_ = (bounds.ub - bounds.lb) / 2
        return (mid_point + range_ * (np.random.rand(self.dim) - 0.5),
                mid_point - range_ * (np.random.rand(self.dim) - 0.5))
    
    def landscape_smoothing(self, func, x, alpha=0.1):
        perturbed = x + alpha * (np.random.rand(self.dim) - 0.5)
        return (func(x) + func(perturbed)) / 2

    def differential_evolution(self, func, bounds, population_size=12, F=0.6, CR=0.95):
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))
        best_solution = None
        best_value = float('inf')
        
        while self.func_evals < self.budget:
            for i in range(population_size):
                a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                CR_dynamic = CR * (1 - self.func_evals / self.budget)
                cross_points = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                
                trial_vector = np.round(trial_vector)
                
                trial_value = self.landscape_smoothing(func, trial_vector)
                self.func_evals += 1
                if trial_value < best_value:
                    best_value = trial_value
                    best_solution = trial_vector
                
                if trial_value < func(population[i]):
                    population[i] = trial_vector
                
                if self.func_evals >= self.budget:
                    break
        
        return best_solution, best_value

    def local_bfgs_optimization(self, func, x0, bounds):
        x0 = 0.5 * (x0 + (bounds.ub + bounds.lb) / 2)
        res = minimize(func, x0, method='L-BFGS-B', bounds=np.array(list(zip(bounds.lb, bounds.ub))))
        return res.x, res.fun

    def __call__(self, func):
        bounds = func.bounds
        init1, init2 = self.quasi_oppositional_initialization(bounds)
        
        best_solution_de, best_value_de = self.differential_evolution(func, bounds)

        if self.func_evals < 0.8 * self.budget:
            best_solution_bfgs, best_value_bfgs = self.local_bfgs_optimization(func, best_solution_de, bounds)
        else:
            best_solution_bfgs, best_value_bfgs = best_solution_de, best_value_de

        return best_solution_bfgs, best_value_bfgs