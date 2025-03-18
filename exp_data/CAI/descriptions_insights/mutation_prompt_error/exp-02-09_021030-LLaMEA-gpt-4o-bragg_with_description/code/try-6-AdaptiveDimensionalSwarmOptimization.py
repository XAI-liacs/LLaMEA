import numpy as np

class AdaptiveDimensionalSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 20
        self.positions = None
        self.velocities = None
        self.best_positions = None
        self.global_best_position = None
        self.inertia_weight = 0.7
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5

    def initialize_swarm(self, bounds):
        self.positions = np.random.uniform(bounds.lb, bounds.ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        self.best_positions = np.copy(self.positions)
        self.global_best_position = self.positions[np.random.randint(self.swarm_size)]

    def __call__(self, func):
        bounds = func.bounds
        self.initialize_swarm(bounds)
        evaluations = 0
        best_score = float('-inf')

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                score = func(self.positions[i])
                evaluations += 1

                if score > func(self.best_positions[i]):
                    self.best_positions[i] = self.positions[i]

                if score > best_score:
                    best_score = score
                    self.global_best_position = self.positions[i]

            self.update_velocities()
            self.update_positions(bounds)

            # Adaptation: Adjust swarm size and inertia weight dynamically
            self.swarm_size = max(5, self.swarm_size - int(self.budget / 1000))
            self.inertia_weight = max(0.1, self.inertia_weight * np.exp(-0.01 * evaluations))  # Nonlinear decay
            self.cognitive_coeff = 2.0 - 1.5 * (evaluations / self.budget)  # Adaptive cognitive coefficient

        return self.global_best_position

    def update_velocities(self):
        for i in range(self.swarm_size):
            cognitive_component = self.cognitive_coeff * np.random.rand(self.dim) * (self.best_positions[i] - self.positions[i])
            social_component = self.social_coeff * np.random.rand(self.dim) * (self.global_best_position - self.positions[i])
            self.velocities[i] = self.inertia_weight * self.velocities[i] + cognitive_component + social_component

    def update_positions(self, bounds):
        self.positions += self.velocities
        self.positions = np.clip(self.positions, bounds.lb, bounds.ub)