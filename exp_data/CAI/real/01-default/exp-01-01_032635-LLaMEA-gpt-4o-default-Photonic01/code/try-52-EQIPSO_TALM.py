import numpy as np

class EQIPSO_TALM:
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
        self.adapt_rate = 0.1
        self.bounds = None
        self.learning_rate = 0.1  # Initial learning rate
        self.memory = np.full((self.swarm_size, self.dim), np.inf)

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim)
        new_position = position + beta * (global_best - position) + delta * 0.1
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def dynamic_inertia_adjustment(self):
        diversity = np.mean(np.std(self.positions, axis=0))
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.4 + 0.5 * (diversity / max_diversity)

    def temporal_adaptive_learning_strategy(self, iteration, max_iterations):
        factor = iteration / max_iterations
        self.learning_rate = 0.1 * (1 - factor)
        for i in range(self.swarm_size):
            if np.random.rand() < self.learning_rate:
                self.velocities[i] += np.random.uniform(-0.1, 0.1, self.dim)

    def update_memory(self, i, current_position):
        self.memory[i] = np.where(current_position < self.memory[i], current_position, self.memory[i])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0
        iteration = 0
        max_iterations = self.budget // self.swarm_size

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

                self.update_memory(i, self.positions[i])

            self.dynamic_inertia_adjustment()
            self.temporal_adaptive_learning_strategy(iteration, max_iterations)

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

            iteration += 1

        return self.global_best_position, self.global_best_value