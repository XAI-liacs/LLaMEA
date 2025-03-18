import numpy as np

class ACPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.clusters = 5
        self.positions = np.random.rand(self.population_size, self.dim)
        self.velocities = np.random.rand(self.population_size, self.dim) * 0.1
        self.personal_best_positions = self.positions.copy()
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.w = 0.7   # Inertia weight

    def __call__(self, func):
        bounds = func.bounds
        lb, ub = bounds.lb, bounds.ub
        evaluations = 0

        while evaluations < self.budget:
            # Evaluate the population
            for i in range(self.population_size):
                if evaluations >= self.budget:
                    break

                score = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i].copy()

                # Update global best
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i].copy()

            # Clustering based on performance
            sorted_indices = np.argsort(self.personal_best_scores)
            clusters = np.array_split(sorted_indices, self.clusters)

            # Update velocities and positions for each cluster
            for cluster in clusters:
                if len(cluster) == 0:
                    continue

                cluster_leader = self.personal_best_positions[cluster[0]]
                for i in cluster:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    cognitive_velocity = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                    social_velocity = self.c2 * r2 * (cluster_leader - self.positions[i])
                    self.velocities[i] = self.w * self.velocities[i] + cognitive_velocity + social_velocity

                    # Update positions
                    self.positions[i] += self.velocities[i]
                    self.positions[i] = np.clip(self.positions[i], lb, ub)

        return self.global_best_position, self.global_best_score