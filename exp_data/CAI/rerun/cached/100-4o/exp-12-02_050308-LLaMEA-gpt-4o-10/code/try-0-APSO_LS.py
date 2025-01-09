import numpy as np

class APSO_LS:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.swarm_size = 40
        self.inertia_weight = 0.7
        self.cognitive_coef = 1.5
        self.social_coef = 1.5
        self.local_search_radius = 0.1
        self.evaluations = 0

        # Initialize particles
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                # Evaluate fitness
                fitness = func(self.positions[i])
                self.evaluations += 1

                # Update personal best
                if fitness < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                # Update global best
                if fitness < self.global_best_score:
                    self.global_best_score = fitness
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                cognitive_velocity = self.cognitive_coef * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_velocity = self.social_coef * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_velocity + social_velocity)
                self.positions[i] += self.velocities[i]

                # Clamp positions to search space
                self.positions[i] = np.clip(self.positions[i], self.lower_bound, self.upper_bound)

            # Perform local search around the global best position
            if self.evaluations < self.budget:
                local_search_position = self.global_best_position + np.random.uniform(
                    -self.local_search_radius, self.local_search_radius, self.dim)
                local_search_position = np.clip(local_search_position, self.lower_bound, self.upper_bound)
                local_search_fitness = func(local_search_position)
                self.evaluations += 1

                if local_search_fitness < self.global_best_score:
                    self.global_best_score = local_search_fitness
                    self.global_best_position = local_search_position

        return self.global_best_position, self.global_best_score