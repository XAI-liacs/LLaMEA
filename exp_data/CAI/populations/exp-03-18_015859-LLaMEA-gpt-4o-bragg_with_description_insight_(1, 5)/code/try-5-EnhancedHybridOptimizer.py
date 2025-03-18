import numpy as np
from scipy.optimize import minimize

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.func_evals = 0

    def quasi_oppositional_initialization(self, bounds):
        mid_point = (bounds.ub + bounds.lb) / 2
        range_ = (bounds.ub - bounds.lb) / 2
        return (mid_point + range_ * (np.random.rand(self.dim) - 0.5),
                mid_point - range_ * (np.random.rand(self.dim) - 0.5))

    def adaptive_differential_evolution(self, func, bounds, population_size=12, F=0.5, CR=0.9):
        population = np.random.uniform(bounds.lb, bounds.ub, (population_size, self.dim))
        best_value = float('inf')
        best_solution = population[0]

        while self.func_evals < self.budget:
            for i in range(population_size):
                a, b, c = population[np.random.choice(population_size, 3, replace=False)]
                F = np.random.uniform(0.4, 0.9)  # Adaptive mutation factor
                mutant_vector = np.clip(a + F * (b - c), bounds.lb, bounds.ub)
                cross_points = np.random.rand(self.dim) < CR
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

    def learning_based_local_bfgs(self, func, x0, bounds):
        memory = [x0]
        def enhanced_func(x):
            memory.append(x)
            return func(x)
        
        res = minimize(enhanced_func, x0, method='L-BFGS-B', bounds=np.array(list(zip(bounds.lb, bounds.ub))))
        return res.x if res.success else memory[-1], res.fun

    def __call__(self, func):
        bounds = func.bounds
        init1, init2 = self.quasi_oppositional_initialization(bounds)
        
        # Step 1: Perform Adaptive Differential Evolution for global exploration
        best_solution_de, best_value_de = self.adaptive_differential_evolution(func, bounds)

        # Step 2: Perform Learning-based Local Optimization using BFGS
        best_solution_bfgs, best_value_bfgs = self.learning_based_local_bfgs(func, best_solution_de, bounds)

        return best_solution_bfgs, best_value_bfgs