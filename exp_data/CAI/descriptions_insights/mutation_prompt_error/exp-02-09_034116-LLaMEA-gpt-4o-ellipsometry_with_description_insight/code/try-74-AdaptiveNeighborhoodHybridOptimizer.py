import numpy as np
from scipy.optimize import minimize
from scipy.stats.qmc import Sobol

class AdaptiveNeighborhoodHybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        best_solution = None
        best_value = float('inf')

        # Exploration phase: Sobol sequence sampling
        sobol_sampler = Sobol(d=self.dim, scramble=True)
        num_samples = max(1, int(np.log2(self.budget)))
        samples = sobol_sampler.random_base2(m=int(np.log2(num_samples)))
        top_k = min(len(samples), int(self.budget * 0.1))  # Increased top-k percentage for broader sampling
        top_samples = sorted(samples[:top_k], key=lambda s: func(lb + s * (ub - lb)))

        # Dynamic neighborhood search
        def dynamic_neighborhood(center, scale=0.1):
            return np.clip(center + np.random.uniform(-scale, scale, size=self.dim), 0, 1)

        # Exploitation phase: Local optimization with dynamic neighborhoods
        for sample in top_samples:
            sample_scaled = lb + sample * (ub - lb)
            neighborhood_scale = 0.1
            neighborhood_best_value = float('inf')
            neighborhood_best_solution = None

            for _ in range(10):  # Dynamic neighborhood exploration
                neighbor = dynamic_neighborhood(sample)
                neighbor_scaled = lb + neighbor * (ub - lb)
                
                def wrapped_func(x):
                    return func(x)
                
                result = minimize(wrapped_func, neighbor_scaled, method='L-BFGS-B', bounds=list(zip(lb, ub)), options={'maxiter': (self.budget // top_k) - num_samples})

                if result.fun < neighborhood_best_value:
                    neighborhood_best_value = result.fun
                    neighborhood_best_solution = result.x

            if neighborhood_best_value < best_value:
                best_value = neighborhood_best_value
                best_solution = neighborhood_best_solution

        return best_solution