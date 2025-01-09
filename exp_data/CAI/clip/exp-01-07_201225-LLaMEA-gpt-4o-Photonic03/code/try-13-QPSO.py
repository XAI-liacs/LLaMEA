import numpy as np

class QPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(30, 5 + dim * 2)  # Adaptive swarm size based on dimensionality
        self.positions = np.random.rand(self.swarm_size, dim)
        self.velocities = np.zeros((self.swarm_size, dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_fitness = np.full(self.swarm_size, np.inf)
        self.global_best_position = None
        self.global_best_fitness = np.inf
        self.inertia_weight = 0.9  # Initial inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient

    def __call__(self, func):
        bounds = np.vstack((func.bounds.lb, func.bounds.ub)).T
        evaluations = 0

        def decode_position(position):
            return bounds[:, 0] + ((bounds[:, 1] - bounds[:, 0]) * position)

        def evaluate_position(decoded_position):
            nonlocal evaluations
            fitness = np.array([func(ind) for ind in decoded_position])
            evaluations += len(decoded_position)
            return fitness

        while evaluations < self.budget:
            decoded_positions = decode_position(self.positions)
            fitness = evaluate_position(decoded_positions)

            for i in range(self.swarm_size):
                if fitness[i] < self.personal_best_fitness[i]:
                    self.personal_best_fitness[i] = fitness[i]
                    self.personal_best_positions[i] = self.positions[i]
                    if fitness[i] < self.global_best_fitness:
                        self.global_best_fitness = fitness[i]
                        self.global_best_position = self.positions[i]

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_part = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_part = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_part + social_part
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], 0, 1)

            # Update inertia weight dynamically
            self.inertia_weight = 0.4 + 0.5 * ((self.budget - evaluations) / self.budget)

        best_solution = decode_position(self.global_best_position)
        return best_solution, self.global_best_fitness