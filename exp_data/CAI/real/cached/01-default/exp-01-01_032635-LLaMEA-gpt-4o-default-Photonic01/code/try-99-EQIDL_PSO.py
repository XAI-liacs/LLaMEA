import numpy as np

class EQIDL_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_position = None
        self.global_best_value = np.inf
        self.w = 0.9  # Initial inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.c3 = 1.0  # Cooperative coefficient
        self.adapt_rate = 0.1
        self.history = []  # Store historical best values
        self.learning_rate = 0.5  # Initial learning rate

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * 0.05  # Exploration boost factor
        new_position = position + beta * (global_best - position) + delta
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_dynamic_inertia(self):
        diversity = np.std(self.positions, axis=0).mean()
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.4 + 0.5 * (diversity / max_diversity)
        if len(self.history) > 1:
            self.w += 0.1 * (self.history[-1] - self.history[-2]) / max(self.history)
        self.learning_rate = min(0.8, max(0.2, diversity / max_diversity))  # Adjust learning rate

    def elite_learning(self, lb, ub):
        elite_count = int(0.1 * self.swarm_size)
        elite_indices = np.argsort(self.personal_best_values)[:elite_count]
        for idx in elite_indices:
            neighborhood_radius = np.exp(-self.dim / self.swarm_size)
            perturbation = np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim)
            self.personal_best_positions[idx] = np.clip(self.personal_best_positions[idx] + perturbation, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.positions[i])
                evaluations += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i].copy()

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.positions[i].copy()

            self.history.append(self.global_best_value)
            self.adaptive_dynamic_inertia()
            self.elite_learning(lb, ub)

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                r3 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                cooperative_component = self.c3 * r3 * (np.mean(self.personal_best_positions, axis=0) - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + cooperative_component
                self.positions[i] += self.learning_rate * self.velocities[i]

                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.quantum_position_update(self.positions[i], self.global_best_position)

        return self.global_best_position, self.global_best_value