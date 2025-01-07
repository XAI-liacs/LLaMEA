import numpy as np

class EQIDL_PSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.num_swarms = 3  # Multi-swarm approach
        self.positions = None
        self.velocities = None
        self.personal_best_positions = None
        self.personal_best_values = None
        self.global_best_positions = [None] * self.num_swarms
        self.global_best_values = [np.inf] * self.num_swarms
        self.w = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.c3 = 1.0
        self.adapt_rate = 0.1
        self.history = []

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.num_swarms, self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.num_swarms, self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full((self.num_swarms, self.swarm_size), np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * 0.05
        new_position = position + beta * (global_best - position) + delta
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_boundary_adjustment(self, lb, ub):
        span = ub - lb
        min_diversity = 0.1 * np.sqrt(np.sum(span**2))  # 10% of the max possible diversity
        for swarm_idx, global_best in enumerate(self.global_best_positions):
            if global_best is not None:
                diversity = np.std(self.positions[swarm_idx], axis=0).mean()
                if diversity < min_diversity:
                    adjust_factor = 0.1 * span
                    new_lb = np.maximum(lb, global_best - adjust_factor)
                    new_ub = np.minimum(ub, global_best + adjust_factor)
                    for i in range(self.swarm_size):
                        if np.any(self.positions[swarm_idx][i] < new_lb) or np.any(self.positions[swarm_idx][i] > new_ub):
                            self.positions[swarm_idx][i] = np.random.uniform(new_lb, new_ub, self.dim)

    def elite_learning(self, lb, ub):
        elite_count = int(0.1 * self.swarm_size)
        for swarm_idx in range(self.num_swarms):
            elite_indices = np.argsort(self.personal_best_values[swarm_idx])[:elite_count]
            for idx in elite_indices:
                neighborhood_radius = np.exp(-self.dim / self.swarm_size)
                perturbation = np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim)
                self.personal_best_positions[swarm_idx][idx] = np.clip(self.personal_best_positions[swarm_idx][idx] + perturbation, lb, ub)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0

        while evaluations < self.budget:
            for swarm_idx in range(self.num_swarms):
                for i in range(self.swarm_size):
                    if evaluations >= self.budget:
                        break

                    current_value = func(self.positions[swarm_idx][i])
                    evaluations += 1

                    if current_value < self.personal_best_values[swarm_idx][i]:
                        self.personal_best_values[swarm_idx][i] = current_value
                        self.personal_best_positions[swarm_idx][i] = self.positions[swarm_idx][i].copy()

                    if current_value < self.global_best_values[swarm_idx]:
                        self.global_best_values[swarm_idx] = current_value
                        self.global_best_positions[swarm_idx] = self.positions[swarm_idx][i].copy()

                self.history.append(min(self.global_best_values))
                self.adaptive_boundary_adjustment(lb, ub)
                self.elite_learning(lb, ub)

                for i in range(self.swarm_size):
                    r1 = np.random.rand(self.dim)
                    r2 = np.random.rand(self.dim)
                    r3 = np.random.rand(self.dim)
                    cognitive_component = self.c1 * r1 * (self.personal_best_positions[swarm_idx][i] - self.positions[swarm_idx][i])
                    social_component = self.c2 * r2 * (self.global_best_positions[swarm_idx] - self.positions[swarm_idx][i])
                    cooperative_component = self.c3 * r3 * (np.mean(self.personal_best_positions[swarm_idx], axis=0) - self.positions[swarm_idx][i])
                    self.velocities[swarm_idx][i] = self.w * self.velocities[swarm_idx][i] + cognitive_component + social_component + cooperative_component
                    self.positions[swarm_idx][i] += self.velocities[swarm_idx][i]

                    if np.random.rand() < self.adapt_rate:
                        self.positions[swarm_idx][i] = self.quantum_position_update(self.positions[swarm_idx][i], self.global_best_positions[swarm_idx])

        global_best_idx = np.argmin(self.global_best_values)
        return self.global_best_positions[global_best_idx], self.global_best_values[global_best_idx]