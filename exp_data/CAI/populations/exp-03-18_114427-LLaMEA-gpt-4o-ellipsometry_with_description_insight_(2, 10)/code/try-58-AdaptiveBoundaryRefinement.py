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

        # Improved centroid selection with confidence interval approach
        confidence_evals = np.array([func(centroid) for centroid in centroids])
        best_index = np.argmin(confidence_evals + np.std(confidence_evals))
        best_solution = centroids[best_index]
        best_value = confidence_evals[best_index]
        remaining_budget -= len(centroids)

        # Enhanced particle swarm step with adaptive sampling
        pso_count = max(3, int(0.1 * remaining_budget))
        pso_samples = np.random.uniform(func.bounds.lb, func.bounds.ub, (pso_count, self.dim))
        pso_evals = [func(sample) for sample in pso_samples]
        remaining_budget -= pso_count
        pso_best_index = np.argmin(pso_evals)
        pso_best_solution = pso_samples[pso_best_index]

        if pso_evals[pso_best_index] < best_value:
            best_solution = pso_best_solution
            best_value = pso_evals[pso_best_index]

        # Iteratively refine solution using dynamic local optimizer selection
        while remaining_budget > 0:
            local_optimizer = 'BFGS' if remaining_budget > 15 else 'Nelder-Mead'
            options = {'maxiter': min(remaining_budget, 50)}

            # Perform local optimization
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)) if local_optimizer != 'BFGS' else None,
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