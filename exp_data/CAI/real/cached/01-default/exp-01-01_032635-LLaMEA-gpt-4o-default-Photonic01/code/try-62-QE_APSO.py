import numpy as np

class QE_APSO:
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
        self.w_max = 0.9  # Max inertia weight
        self.w_min = 0.4  # Min inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.adapt_rate = 0.15
        self.bounds = None
        self.tunneling_intensity = 0.1  # Tunneling intensity factor
        self.diversity_threshold = 0.1  # Diversity threshold for adaptive strategy

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.global_best_value = np.inf
        self.bounds = (lb, ub)

    def quantum_tunneling(self, position, global_best):
        direction = np.random.normal(0, 1, self.dim)
        intensity = self.tunneling_intensity * np.random.exponential(1.0)
        new_position = position + direction * intensity * (global_best - position)
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def diversity_control(self):
        current_diversity = np.mean(np.std(self.positions, axis=0))
        if current_diversity < self.diversity_threshold:
            self.velocities += np.random.uniform(-0.5, 0.5, self.velocities.shape)

    def update_inertia_weight(self, evaluations):
        progress = evaluations / self.budget
        self.w = self.w_max - (self.w_max - self.w_min) * progress

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

            self.update_inertia_weight(evaluations)
            self.diversity_control()

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Quantum-inspired tunneling
                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.quantum_tunneling(self.positions[i], self.global_best_position)

        return self.global_best_position, self.global_best_value