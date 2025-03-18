import numpy as np
from scipy.optimize import minimize
from scipy.optimize import differential_evolution

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        
    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        remaining_budget = self.budget
        
        # Initial exploration with enhanced sampling
        num_samples = min(remaining_budget // 4, 25 * self.dim)
        samples = np.random.uniform(lb, ub, (num_samples, self.dim))
        remaining_budget -= num_samples
        
        sample_evaluations = np.array([func(sample) for sample in samples])
        best_idx = np.argmin(sample_evaluations)
        best_solution = samples[best_idx]
        best_value = sample_evaluations[best_idx]
        
        # Particle Swarm exploration
        def particle_swarm_objective(x):
            return func(x)
        
        if remaining_budget > 10:
            bounds = [(lb[i], ub[i]) for i in range(self.dim)]
            result = differential_evolution(particle_swarm_objective, bounds, maxiter=5, popsize=10, strategy='best1bin')
            if result.fun < best_value:
                best_solution = result.x
                best_value = result.fun
            remaining_budget -= 50  # Deduct approximate budget used
        
        # Local optimization using Nelder-Mead
        def local_objective(x):
            return func(x)

        remaining_budget = max(remaining_budget, 5)
        result = minimize(local_objective, best_solution, bounds=list(zip(lb, ub)), method='Nelder-Mead', options={'maxfev': remaining_budget})
        
        if result.fun < best_value:
            best_solution = result.x
            best_value = result.fun
        
        return best_solution