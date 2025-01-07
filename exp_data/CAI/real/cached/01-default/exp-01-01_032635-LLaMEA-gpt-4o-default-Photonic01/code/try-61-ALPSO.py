import numpy as np

class ALPSO:
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
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.alpha = 1.5  # Levy flight parameter
        self.bounds = None

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def levy_flight(self, position):
        step = np.random.standard_normal(self.dim) * self.levy()
        new_position = position + step
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def levy(self):
        u = np.random.standard_normal(self.dim) * (np.sqrt(np.pi) / 2) ** (1 / self.alpha)
        v = np.random.standard_normal(self.dim)
        z = u / (abs(v) ** (1 / self.alpha))
        return z

    def adaptive_inertia_weight(self):
        diversity = np.mean(np.std(self.positions, axis=0))
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.5 + 0.4 * (diversity / max_diversity)

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

            self.adaptive_inertia_weight()

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Levy flight for enhanced exploration
                if np.random.rand() < 0.3:
                    self.positions[i] = self.levy_flight(self.positions[i])

        return self.global_best_position, self.global_best_value