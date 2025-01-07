import numpy as np

class QE_PSO:
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
        self.w = 0.7  # Initial inertia weight
        self.c1 = 2.0  # Cognitive coefficient
        self.c2 = 2.0  # Social coefficient
        self.mutation_rate = 0.1
        self.history = []  # Store historical best values

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def quantum_position_update(self, position, global_best):
        alpha = np.random.uniform(-1, 1, self.dim)
        delta = np.random.normal(0, 0.1, self.dim)  # Quantum fluctuation factor
        new_position = position + alpha * (global_best - position) + delta
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def mutate_positions(self, positions, rate):
        for i in range(self.swarm_size):
            if np.random.rand() < rate:
                mutation_vector = np.random.uniform(-0.1, 0.1, self.dim)
                positions[i] = np.clip(positions[i] + mutation_vector, *self.bounds)

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

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                if np.random.rand() < self.mutation_rate:
                    self.positions[i] = self.quantum_position_update(self.positions[i], self.global_best_position)

            # Apply evolutionary selection
            selected_indices = np.argsort(self.personal_best_values)[:self.swarm_size // 2]
            self.positions = self.positions[selected_indices]
            self.velocities = self.velocities[selected_indices]
            self.mutate_positions(self.positions, self.mutation_rate)

        return self.global_best_position, self.global_best_value