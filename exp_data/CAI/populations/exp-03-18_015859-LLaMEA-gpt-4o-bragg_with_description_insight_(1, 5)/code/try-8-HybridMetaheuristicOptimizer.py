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

    def differential_evolution(self, func, bounds, F=0.5, CR=0.9):
        population_size = int(10 + (self.budget - self.func_evals) / 10)  # Adaptive population size
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))
        best_solution = None
        best_value = float('inf')
        
        while self.func_evals < self.budget:
            for i in range(population_size):
                a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                CR_dynamic = CR * (1 - self.func_evals / self.budget)  # Adaptive crossover rate
                cross_points = np.random.rand(self.dim) < CR_dynamic
                trial_vector = np.where(cross_points, mutant_vector, population[i])
                
                trial_value = func(trial_vector)
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
        x0 = 0.5 * (x0 + (bounds.ub + bounds.lb) / 2)  # Strategically adjust initial x0
        res = minimize(func, x0, method='L-BFGS-B', bounds=np.array(list(zip(bounds.lb, bounds.ub))))
        return res.x, res.fun

    def __call__(self, func):
        bounds = func.bounds
        init1, init2 = self.quasi_oppositional_initialization(bounds)
        
        # Step 1: Perform Differential Evolution for global exploration
        best_solution_de, best_value_de = self.differential_evolution(func, bounds)

        # Step 2: Perform local optimization using BFGS from best DE solution
        best_solution_bfgs, best_value_bfgs = self.local_bfgs_optimization(func, best_solution_de, bounds)

        # Return the best found solution and its value
        return best_solution_bfgs, best_value_bfgs