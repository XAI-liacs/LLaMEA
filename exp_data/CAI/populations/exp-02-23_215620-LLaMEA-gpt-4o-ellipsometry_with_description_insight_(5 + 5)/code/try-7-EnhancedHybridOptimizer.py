import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import LatinHypercube

class EnhancedHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.evals = 0
    
    def __call__(self, func):
        # Define bounds
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Latin Hypercube Sampling for initial exploration
        lhs = LatinHypercube(d=self.dim, seed=42)
        initial_points = lhs.random(int(self.budget * 0.2))
        initial_guesses = lb + (ub - lb) * initial_points
        
        # Initial sampling
        best_guess = initial_guesses[0]
        best_value = func(best_guess)
        self.evals += 1

        for i in range(1, len(initial_guesses)):
            if self.evals >= self.budget:
                break
            guess = initial_guesses[i]
            value = func(guess)
            self.evals += 1
            if value < best_value:
                best_value = value
                best_guess = guess

        # Adaptive adjustment of bounds based on initial search
        if self.evals < self.budget * 0.4:
            # Compute sample median to refine search area
            sample_median = np.median(initial_guesses, axis=0)
            lb = np.maximum(lb, sample_median - (sample_median - lb) * 0.5)
            ub = np.minimum(ub, sample_median + (ub - sample_median) * 0.5)
        
        # Use BFGS for local optimization
        options = {'maxiter': self.budget - self.evals}
        result = minimize(func, best_guess, method='BFGS', bounds=list(zip(lb, ub)), options=options)
        
        return result.x if result.success else best_guess