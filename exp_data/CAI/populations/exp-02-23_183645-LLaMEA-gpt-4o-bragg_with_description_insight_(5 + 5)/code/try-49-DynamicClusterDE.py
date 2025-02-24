import numpy as np
from scipy.optimize import minimize
from sklearn.cluster import KMeans

class DynamicClusterDE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.cross_prob = 0.7
        self.diff_weight = 0.8
        self.local_optimization_prob = 0.1
        self.cluster_merge_threshold = 1e-3

    def _initialize_population(self, lb, ub):
        mid_point = (lb + ub) / 2
        half_range = (ub - lb) / 2
        return mid_point + np.random.uniform(-half_range, half_range, (self.population_size, self.dim))

    def _evaluate_population(self, population, func):
        return np.array([func(ind) for ind in population])

    def _differential_evolution_step(self, population, scores, lb, ub, func):
        for i in range(self.population_size):
            indices = [idx for idx in range(self.population_size) if idx != i]
            a, b, c = np.random.choice(indices, 3, replace=False)
            
            mutant = np.clip(population[a] + self.diff_weight * (population[b] - population[c]), lb, ub)
            cross_points = np.random.rand(self.dim) < self.cross_prob
            if not np.any(cross_points):
                cross_points[np.random.randint(0, self.dim)] = True
            trial = np.where(cross_points, mutant, population[i])
            
            trial_score = func(trial)
            if trial_score > scores[i]:
                population[i] = trial
                scores[i] = trial_score

        return population, scores

    def _local_optimization(self, candidate, func, lb, ub):
        def local_func(x):
            return -func(x)

        result = minimize(local_func, candidate, bounds=[(lb[j], ub[j]) for j in range(self.dim)],
                          method='L-BFGS-B', options={'maxiter': 10})
        return result.x if result.success else candidate

    def _dynamic_clustering(self, population):
        kmeans = KMeans(n_clusters=min(5, len(population)))
        kmeans.fit(population)
        return kmeans.cluster_centers_

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        population = self._initialize_population(lb, ub)
        scores = self._evaluate_population(population, func)
        
        evaluations = self.population_size
        while evaluations < self.budget:
            population, scores = self._differential_evolution_step(population, scores, lb, ub, func)
            evaluations += self.population_size

            cluster_centers = self._dynamic_clustering(population)
            if len(cluster_centers) > 1:
                distances = np.linalg.norm(cluster_centers[None, :] - cluster_centers[:, None], axis=2)
                min_distance = np.min(distances + np.eye(len(cluster_centers)) * np.inf)
                if min_distance < self.cluster_merge_threshold:
                    population = cluster_centers
                    scores = self._evaluate_population(population, func)
                    evaluations += len(cluster_centers)
            
            if np.random.rand() < self.local_optimization_prob:
                idx = np.random.randint(0, self.population_size)
                candidate = population[idx]
                optimized = self._local_optimization(candidate, func, lb, ub)
                optimized_score = func(optimized)
                evaluations += 1
                if optimized_score > scores[idx]:
                    population[idx] = optimized
                    scores[idx] = optimized_score
        
        best_idx = np.argmax(scores)
        return population[best_idx]