import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

class HybridOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        # Step 1: Initial uniform sampling with dynamically adjusted sample size
        sample_size = max(5, int(self.budget // 2.5))
        bounds = np.array([func.bounds.lb, func.bounds.ub]).T
        samples = np.random.uniform(bounds[:, 0], bounds[:, 1], (sample_size, self.dim))

        # Evaluate the initial samples
        sample_costs = [func(sample) for sample in samples]
        self.budget -= sample_size

        # Step 2: Strategic clustering for better initial sample selection
        num_clusters = min(5, len(samples))  # Create clusters based on sample size
        kmeans = KMeans(n_clusters=num_clusters).fit(samples)
        cluster_centers = kmeans.cluster_centers_
        cluster_costs = [func(center) for center in cluster_centers]
        best_cluster_index = np.argmin(cluster_costs)
        best_sample = cluster_centers[best_cluster_index]
        best_cost = cluster_costs[best_cluster_index]

        # Step 3: Adaptive bounds adjustment
        adaptive_bounds = bounds.copy()
        for i in range(self.dim):
            span = (adaptive_bounds[i, 1] - adaptive_bounds[i, 0]) * 0.12
            adaptive_bounds[i, 0] = max(func.bounds.lb[i], best_sample[i] - span)
            adaptive_bounds[i, 1] = min(func.bounds.ub[i], best_sample[i] + span)

        # Step 4: Local optimization using BFGS within adjusted bounds
        def bounded_func(x):
            if np.all(x >= func.bounds.lb) and np.all(x <= func.bounds.ub):
                return func(x)
            return np.inf

        result = minimize(bounded_func, best_sample, method='BFGS', bounds=adaptive_bounds, options={'maxfun': self.budget})
        
        # Return the best-found solution
        return result.x if result.success else best_sample