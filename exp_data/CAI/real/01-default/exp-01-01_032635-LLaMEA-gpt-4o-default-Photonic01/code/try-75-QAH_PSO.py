import numpy as np

class QAH_PSO:
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
        self.w = 0.7  # Adjusted inertia weight for a better balance
        self.c1 = 1.4  # Cognitive coefficient
        self.c2 = 1.6  # Social coefficient
        self.c3 = 1.2  # Hierarchical cooperative coefficient
        self.adapt_rate = 0.2
        self.exploration_boost = 0.1  # Increased exploration boost
        self.elite_fraction = 0.15  # Increased fraction of elite particles
        self.hierarchy_levels = 3

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * self.exploration_boost
        new_position = position + beta * (global_best - position) + delta
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def adaptive_neighborhood_search(self, position, lb, ub):
        neighborhood_radius = np.exp(-self.dim / self.swarm_size)
        perturbation = np.random.uniform(-neighborhood_radius, neighborhood_radius, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, lb, ub)

    def dynamic_inertia_adjustment(self):
        diversity = np.mean(np.std(self.positions, axis=0))
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.5 + 0.4 * (diversity / max_diversity)

    def hierarchical_cooperation(self):
        level_count = int(self.swarm_size / self.hierarchy_levels)
        for level in range(self.hierarchy_levels):
            start_index = level * level_count
            end_index = min((level + 1) * level_count, self.swarm_size)
            local_best_value = np.inf
            local_best_position = None
            for i in range(start_index, end_index):
                if self.personal_best_values[i] < local_best_value:
                    local_best_value = self.personal_best_values[i]
                    local_best_position = self.personal_best_positions[i].copy()
            for i in range(start_index, end_index):
                r3 = np.random.rand(self.dim)
                positional_hierarchy_component = self.c3 * r3 * (local_best_position - self.positions[i])
                self.velocities[i] += positional_hierarchy_component

    def elite_learning(self, lb, ub):
        elite_count = int(self.elite_fraction * self.swarm_size)
        elite_indices = np.argsort(self.personal_best_values)[:elite_count]
        for idx in elite_indices:
            self.personal_best_positions[idx] = self.adaptive_neighborhood_search(self.personal_best_positions[idx], lb, ub)

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

            self.dynamic_inertia_adjustment()
            self.hierarchical_cooperation()
            self.elite_learning(lb, ub)

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Quantum-inspired position update
                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.quantum_position_update(self.positions[i], self.global_best_position)

                # Adaptive neighborhood search
                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.adaptive_neighborhood_search(self.positions[i], lb, ub)

        return self.global_best_position, self.global_best_value