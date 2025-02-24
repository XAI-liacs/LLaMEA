import numpy as np
from scipy.stats.qmc import Sobol
from skopt import Optimizer

class BayesianAdaptiveOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Define the search space
        lb = np.array(func.bounds.lb)
        ub = np.array(func.bounds.ub)
        
        # Initialize variables
        current_budget = 0
        best_solution = None
        best_score = float('inf')
        
        # Use Sobol sequence for initial points to improve coverage
        sampler = Sobol(d=self.dim, scramble=True)
        initial_points = lb + (ub - lb) * sampler.random_base2(m=4)
        
        # Setup Bayesian optimization with Gaussian Process
        optimizer = Optimizer(dimensions=[(l, u) for l, u in zip(lb, ub)], base_estimator='GP', n_initial_points=0)
        optimizer.tell(initial_points, [float('inf')] * len(initial_points))
        
        # Evaluate initial points
        for point in initial_points:
            if current_budget >= self.budget:
                break

            score = func(point)
            current_budget += 1
            
            optimizer.tell(point, score)

            # Update the best solution found
            if score < best_score:
                best_solution = point
                best_score = score

        while current_budget < self.budget:
            next_point = optimizer.ask()
            score = func(next_point)
            current_budget += 1
            
            optimizer.tell(next_point, score)

            # Update the best solution found
            if score < best_score:
                best_solution = next_point
                best_score = score

        return best_solution