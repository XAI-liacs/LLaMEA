import numpy as np
from scipy.spatial.distance import cdist

class AHCO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.best_solution = None
        self.best_value = float('inf')
        self.population_size = 30
        self.clusters = []
        self.cluster_assignments = []
    
    def initialize_population(self, lb, ub):
        return lb + (ub - lb) * np.random.rand(self.population_size, self.dim)
    
    def adaptive_hierarchy(self, data, n_clusters):
        distances = cdist(data, data)
        clusters = {i: [i] for i in range(len(data))}
        while len(clusters) > n_clusters:
            # find closest pair of clusters
            min_dist = float('inf')
            merge_pair = ()
            for i in clusters:
                for j in clusters:
                    if i < j:
                        dist = np.min(distances[np.ix_(clusters[i], clusters[j])])
                        if dist < min_dist:
                            min_dist = dist
                            merge_pair = (i, j)
            i, j = merge_pair
            # merge clusters
            clusters[i].extend(clusters[j])
            del clusters[j]
        return list(clusters.values())
    
    def update_clusters(self, population, lb, ub):
        n_clusters = max(2, int(self.population_size * 0.1))
        cluster_indices = self.adaptive_hierarchy(population, n_clusters)
        self.clusters = [population[indices] for indices in cluster_indices]
        self.cluster_assignments = np.zeros(self.population_size, dtype=int)
        for cluster_id, indices in enumerate(cluster_indices):
            for i in indices:
                self.cluster_assignments[i] = cluster_id
    
    def exploit_cluster(self, cluster, func, lb, ub):
        centroid = np.mean(cluster, axis=0)
        for i in range(len(cluster)):
            perturbation = (np.random.rand(self.dim) - 0.5) * 0.1 * (ub - lb)
            candidate = np.clip(centroid + perturbation, lb, ub)
            if func(candidate) < func(cluster[i]):
                cluster[i] = candidate
    
    def __call__(self, func):
        lb, ub = np.array(func.bounds.lb), np.array(func.bounds.ub)
        evaluations = 0
        
        population = self.initialize_population(lb, ub)
        self.update_clusters(population, lb, ub)
        
        while evaluations < self.budget:
            for i, pos in enumerate(population):
                value = func(pos)
                evaluations += 1
                
                if value < self.best_value:
                    self.best_value = value
                    self.best_solution = pos.copy()
                
                if evaluations >= self.budget:
                    break
            
            self.update_clusters(population, lb, ub)
            
            for cluster in self.clusters:
                self.exploit_cluster(cluster, func, lb, ub)
        
        return self.best_solution, self.best_value