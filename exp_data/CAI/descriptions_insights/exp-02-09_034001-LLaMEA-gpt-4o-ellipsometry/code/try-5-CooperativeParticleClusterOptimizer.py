import numpy as np

class CooperativeParticleClusterOptimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        cluster_size = 5 * self.dim  # Heuristic choice for initial cluster size
        num_clusters = 2  # Number of clusters
        inertia_weight = 0.7
        cognitive_coeff = 1.5
        social_coeff = 1.5
        np.random.seed(42)

        # Initialize particle clusters randomly within bounds
        clusters = [np.random.uniform(lb, ub, (cluster_size, self.dim)) for _ in range(num_clusters)]
        velocities = [np.random.uniform(-1, 1, (cluster_size, self.dim)) for _ in range(num_clusters)]
        fitness_clusters = [np.array([func(ind) for ind in cluster]) for cluster in clusters]
        evaluations = cluster_size * num_clusters

        personal_bests = [np.copy(cluster) for cluster in clusters]
        personal_best_fitness = [np.copy(fitness) for fitness in fitness_clusters]

        global_best = None
        global_best_fitness = np.inf

        for i in range(num_clusters):
            min_idx = np.argmin(personal_best_fitness[i])
            if personal_best_fitness[i][min_idx] < global_best_fitness:
                global_best_fitness = personal_best_fitness[i][min_idx]
                global_best = personal_bests[i][min_idx]

        while evaluations < self.budget:
            for i in range(num_clusters):
                # Update velocities and positions
                for j in range(cluster_size):
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i][j] = (inertia_weight * velocities[i][j] +
                                        cognitive_coeff * r1 * (personal_bests[i][j] - clusters[i][j]) +
                                        social_coeff * r2 * (global_best - clusters[i][j]))
                    clusters[i][j] = clusters[i][j] + velocities[i][j]
                    clusters[i][j] = np.clip(clusters[i][j], lb, ub)

                # Evaluate new positions
                for j in range(cluster_size):
                    fitness = func(clusters[i][j])
                    evaluations += 1

                    if fitness < personal_best_fitness[i][j]:
                        personal_bests[i][j] = clusters[i][j]
                        personal_best_fitness[i][j] = fitness

                        if fitness < global_best_fitness:
                            global_best_fitness = fitness
                            global_best = clusters[i][j]

                if evaluations >= self.budget:
                    break

            # Dynamic regrouping phase
            if evaluations % (self.budget // 3) == 0:
                all_particles = np.vstack(clusters)
                all_fitness = np.hstack(personal_best_fitness)
                sorted_indices = np.argsort(all_fitness)
                clusters = [all_particles[sorted_indices[k*cluster_size:(k+1)*cluster_size]] for k in range(num_clusters)]
                fitness_clusters = [all_fitness[sorted_indices[k*cluster_size:(k+1)*cluster_size]] for k in range(num_clusters)]
                personal_bests = [np.copy(cluster) for cluster in clusters]
                personal_best_fitness = [np.copy(fitness) for fitness in fitness_clusters]

        return global_best, global_best_fitness