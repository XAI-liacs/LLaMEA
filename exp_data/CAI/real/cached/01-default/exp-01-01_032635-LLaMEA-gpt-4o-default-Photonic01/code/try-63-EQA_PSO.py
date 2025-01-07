import numpy as np

class EQA_PSO:
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
        self.bounds = None
        self.quantum_chance = 0.2  # Chance to perform quantum tunneling
        self.elite_learning_rate = 0.05  # Learning rate for elite particles
        self.elite_fraction = 0.1  # Fraction of elite particles

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_tunneling(self, position, global_best, lb, ub):
        new_position = (position + global_best) / 2 + np.random.normal(0, 0.1, self.dim)
        return np.clip(new_position, lb, ub)

    def elite_learning(self, position, global_best, lb, ub):
        learning_vector = np.random.uniform(-self.elite_learning_rate, self.elite_learning_rate, self.dim)
        new_position = position + learning_vector * (global_best - position)
        return np.clip(new_position, lb, ub)

    def dynamic_inertia_adjustment(self):
        diversity = np.mean(np.std(self.positions, axis=0))
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.4 + 0.5 * (diversity / max_diversity)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0
        elite_count = int(self.elite_fraction * self.swarm_size)

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

            # Sort particles based on personal best values
            sorted_indices = np.argsort(self.personal_best_values)
            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Quantum-inspired tunneling for non-elite particles
                if i >= elite_count and np.random.rand() < self.quantum_chance:
                    self.positions[i] = self.quantum_tunneling(self.positions[i], self.global_best_position, lb, ub)

                # Elite learning strategy for elite particles
                if i < elite_count:
                    self.positions[i] = self.elite_learning(self.positions[i], self.global_best_position, lb, ub)

        return self.global_best_position, self.global_best_value