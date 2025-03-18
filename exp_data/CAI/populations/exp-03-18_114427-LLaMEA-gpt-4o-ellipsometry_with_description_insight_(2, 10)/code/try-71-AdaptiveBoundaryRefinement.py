import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.spatial import Voronoi

class AdaptiveBoundaryRefinement:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        remaining_budget = self.budget

        # Uniform sampling for the initial broad search
        sampler = qmc.LatinHypercube(d=self.dim)
        random_samples = qmc.scale(sampler.random(10), func.bounds.lb, func.bounds.ub)
        sample_evals = [func(sample) for sample in random_samples]
        remaining_budget -= 10

        # Perform Voronoi partitioning on the samples
        vor = Voronoi(random_samples)
        centroids = np.array([np.mean(vor.vertices[region], axis=0) for region in vor.regions if region and -1 not in region])

        # Choose the best initial guess from the centroids
        centroid_evals = [func(centroid) for centroid in centroids]
        best_index = np.argmin(centroid_evals)
        best_solution = centroids[best_index]
        best_value = centroid_evals[best_index]
        remaining_budget -= len(centroids)

        # Enhanced particle swarm step with adaptive particle count
        pso_count = max(3, int(0.1 * remaining_budget))
        pso_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (pso_count, self.dim))
        pso_evals = [func(sample) for sample in pso_samples]
        remaining_budget -= pso_count
        pso_best_index = np.argmin(pso_evals)
        pso_best_solution = pso_samples[pso_best_index]

        if pso_evals[pso_best_index] < best_value:
            best_solution = pso_best_solution
            best_value = pso_evals[pso_best_index]

        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            # Define a gradient-based local optimization strategy for precision
            local_optimizer = 'L-BFGS-B'  # Changed to L-BFGS-B to allow bounds
            options = {'maxiter': min(remaining_budget, 50), 'disp': True}

            # Perform local optimization
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                options=options
            )
            
            # Update remaining budget and best solution found
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds to be closer to the best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (func.bounds.ub - func.bounds.lb))
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (func.bounds.ub - func.bounds.lb))
            
            # Early stopping if budget is exhausted
            if remaining_budget <= 0:
                break
        
        return best_solution