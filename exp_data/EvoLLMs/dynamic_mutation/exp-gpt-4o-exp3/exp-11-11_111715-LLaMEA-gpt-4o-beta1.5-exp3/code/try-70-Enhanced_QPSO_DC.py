import numpy as np
from sklearn.cluster import KMeans

class Enhanced_QPSO_DC:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 10 + 2 * int(np.sqrt(dim))
        self.inertia_weight = 0.9
        self.cognitive_coeff = 2.0
        self.social_coeff = 2.0
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.damping = 0.99
        self.quantum_prob = 0.1
        self.exploration_factor = 0.5
        self.local_search_radius = 0.1
        self.num_clusters = max(2, int(self.pop_size / 5))  # Dynamically determine number of clusters

    def __call__(self, func):
        np.random.seed(0)

        # Initialize particles
        positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        velocities = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        personal_best_positions = np.copy(positions)
        personal_best_scores = np.array([func(ind) for ind in positions])
        global_best_position = personal_best_positions[np.argmin(personal_best_scores)]
        global_best_score = np.min(personal_best_scores)

        evals = self.pop_size

        while evals < self.budget:
            self.inertia_weight *= self.damping

            # Perform particle regrouping using clustering
            kmeans = KMeans(n_clusters=self.num_clusters, random_state=0).fit(positions)
            cluster_centers = kmeans.cluster_centers_
            cluster_assignments = kmeans.labels_

            for i in range(self.pop_size):
                dynamic_exploration = self.exploration_factor * (1 - evals / self.budget)
                
                if np.random.rand() < self.quantum_prob:
                    cluster_center = cluster_centers[cluster_assignments[i]]
                    quantum_distance = np.abs(global_best_position - cluster_center)
                    positions[i] = cluster_center + quantum_distance * np.random.uniform(-dynamic_exploration, dynamic_exploration, self.dim)
                else:
                    r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                    velocities[i] = (self.inertia_weight * velocities[i]
                                    + self.cognitive_coeff * r1 * (personal_best_positions[i] - positions[i])
                                    + self.social_coeff * r2 * (global_best_position - positions[i]))
                    positions[i] = np.clip(positions[i] + velocities[i], self.lower_bound, self.upper_bound)

                score = func(positions[i])
                evals += 1

                if score < personal_best_scores[i]:
                    personal_best_scores[i] = score
                    personal_best_positions[i] = positions[i]

                if score < global_best_score:
                    global_best_score = score
                    global_best_position = positions[i]

            if evals < self.budget:
                for _ in range(self.pop_size // 2):
                    local_positions = global_best_position + np.random.uniform(-self.local_search_radius, self.local_search_radius, self.dim)
                    local_positions = np.clip(local_positions, self.lower_bound, self.upper_bound)
                    local_score = func(local_positions)
                    evals += 1
                    if local_score < global_best_score:
                        global_best_score = local_score
                        global_best_position = local_positions
                        self.local_search_radius *= 0.9
                    else:
                        self.local_search_radius = min(0.1, self.local_search_radius * 1.1)

        return global_best_score