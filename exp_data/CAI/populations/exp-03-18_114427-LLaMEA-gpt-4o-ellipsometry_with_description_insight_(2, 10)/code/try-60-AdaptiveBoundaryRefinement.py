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
        random_samples = qmc.scale(sampler.random(15), func.bounds.lb, func.bounds.ub)  # Increased initial samples
        sample_evals = [func(sample) for sample in random_samples]
        remaining_budget -= 15

        # Perform Voronoi partitioning on the samples
        vor = Voronoi(random_samples)
        centroids = np.array([np.mean(vor.vertices[region], axis=0) for region in vor.regions if region and -1 not in region])

        # Choose the best initial guess from the centroids
        centroid_evals = [func(centroid) for centroid in centroids]
        best_index = np.argmin(centroid_evals)
        best_solution = centroids[best_index]
        best_value = centroid_evals[best_index]
        remaining_budget -= len(centroids)
        
        # Adaptive Particle Swarm Optimization step
        pso_count = max(5, int(0.15 * remaining_budget))  # Increased particle swarm samples
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
            local_optimizer = 'BFGS' if remaining_budget > 20 else 'Nelder-Mead'  # Adjusted threshold for switching
            options = {'maxiter': min(remaining_budget, 60), 'disp': True}  # Increased max iterations

            # Perform local optimization
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)) if local_optimizer != 'BFGS' else None,
                options=options
            )
            
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x
            
            # Update bounds to be closer to the best solution
            func.bounds.lb = np.maximum(func.bounds.lb, best_solution - 0.05 * (func.bounds.ub - func.bounds.lb))  # Reduced bound adjustment
            func.bounds.ub = np.minimum(func.bounds.ub, best_solution + 0.05 * (func.bounds.ub - func.bounds.lb))
            
            if remaining_budget <= 0:
                break
        
        return best_solution