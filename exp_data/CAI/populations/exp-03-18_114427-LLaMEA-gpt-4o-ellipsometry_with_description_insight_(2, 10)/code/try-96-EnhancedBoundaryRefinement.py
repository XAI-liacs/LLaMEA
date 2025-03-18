import numpy as np
from scipy.optimize import minimize
from scipy.stats import qmc
from scipy.spatial import Voronoi
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF

class EnhancedBoundaryRefinement:
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

        # Integrate Gaussian Process for surrogate modeling
        kernel = RBF(length_scale=1.0)
        gp = GaussianProcessRegressor(kernel=kernel)
        gp.fit(random_samples, sample_evals)

        # Choose the best initial guess from the centroids
        centroid_evals = [func(centroid) for centroid in centroids]
        best_index = np.argmin(centroid_evals)
        best_solution = centroids[best_index]
        best_value = centroid_evals[best_index]
        remaining_budget -= len(centroids)

        # Enhanced dynamic sampling with surrogate-based exploration
        surrogate_samples = gp.sample_y(random_samples, n_samples=5)
        surrogate_evals = [func(sample) for sample in surrogate_samples]
        remaining_budget -= len(surrogate_samples)
        surrogate_best_index = np.argmin(surrogate_evals)
        surrogate_best_solution = surrogate_samples[surrogate_best_index]

        if surrogate_evals[surrogate_best_index] < best_value:
            best_solution = surrogate_best_solution
            best_value = surrogate_evals[surrogate_best_index]

        # Iteratively refine solution using local optimizer
        while remaining_budget > 0:
            local_optimizer = 'L-BFGS-B'
            options = {'maxiter': min(remaining_budget, 50), 'disp': False}
            result = minimize(
                func, best_solution, method=local_optimizer,
                bounds=list(zip(func.bounds.lb, func.bounds.ub)),
                options=options
            )
            
            remaining_budget -= result.nfev
            if result.fun < best_value:
                best_value = result.fun
                best_solution = result.x

            # Ensure the new bounds are consistent and within search space
            new_lb = np.maximum(func.bounds.lb, best_solution - 0.1 * (func.bounds.ub - func.bounds.lb))
            new_ub = np.minimum(func.bounds.ub, best_solution + 0.1 * (func.bounds.ub - func.bounds.lb))
            # Change: Ensure bounds consistency by using np.clip
            func.bounds.lb, func.bounds.ub = np.clip(new_lb, func.bounds.lb, func.bounds.ub), np.clip(new_ub, func.bounds.lb, func.bounds.ub)
            
            if remaining_budget <= 0:
                break
        
        return best_solution