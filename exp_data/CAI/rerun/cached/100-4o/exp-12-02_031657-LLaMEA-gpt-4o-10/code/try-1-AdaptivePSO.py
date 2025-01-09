import numpy as np

class AdaptivePSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20 + int(2 * np.sqrt(dim))  # Dynamic swarm size based on dimension
        self.position = np.random.uniform(-5.0, 5.0, (self.swarm_size, dim))
        self.velocity = np.random.uniform(-1.0, 1.0, (self.swarm_size, dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.swarm_size, np.inf)
        self.global_best_position = np.zeros(dim)
        self.global_best_value = np.inf
        self.inertia_weight = 0.7        # Inertia weight
        self.cognitive_param = 1.5       # Cognitive parameter
        self.social_param = 1.5          # Social parameter
        self.mutation_prob = 0.1         # Mutation probability for diversification
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            for i in range(self.swarm_size):
                if self.evaluations >= self.budget:
                    break

                current_value = func(self.position[i])
                self.evaluations += 1

                # Update personal best
                if current_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = current_value
                    self.personal_best_position[i] = self.position[i].copy()

                # Update global best
                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.position[i].copy()

            # Update positions and velocities
            for i in range(self.swarm_size):
                if np.random.rand() < self.mutation_prob:
                    # Mutation step for exploration
                    self.position[i] = np.random.uniform(-5.0, 5.0, self.dim)
                else:
                    self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                        self.cognitive_param * np.random.rand(self.dim) *
                                        (self.personal_best_position[i] - self.position[i]) +
                                        self.social_param * np.random.rand(self.dim) *
                                        (self.global_best_position - self.position[i]))
                    self.position[i] += self.velocity[i]
                    self.position[i] = np.clip(self.position[i], -5.0, 5.0)

        return self.global_best_position, self.global_best_value