import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol
from pyswarm import pso

class SobolPSOOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Calculate the number of initial samples with Sobol sequence
        initial_sample_count = max(10, self.budget // 10)
        
        # Generate Sobol sequence samples within bounds
        sobol_sampler = Sobol(d=self.dim, seed=0)
        sobol_samples = sobol_sampler.random_base2(m=int(np.log2(initial_sample_count)))
        
        # Scale Sobol samples to fit within bounds
        scaled_samples = []
        for sample in sobol_samples:
            scaled_sample = [
                lb + sample[i] * (ub - lb) for i, (lb, ub) in enumerate(zip(func.bounds.lb, func.bounds.ub))
            ]
            scaled_samples.append(scaled_sample)

        # Evaluate initial samples and find the best one
        best_sample = None
        best_value = float('inf')
        for sample in scaled_samples:
            value = func(sample)
            self.budget -= 1
            if value < best_value:
                best_value = value
                best_sample = sample
            if self.budget <= 0:
                return best_sample

        # Define the objective function for PSO
        def objective(x):
            return func(x)

        # Use Particle Swarm Optimization (PSO) from pyswarm within bounds
        lb = func.bounds.lb
        ub = func.bounds.ub
        xopt, fopt = pso(objective, lb, ub, swarmsize=min(20, self.budget), maxiter=self.budget//10)

        return xopt