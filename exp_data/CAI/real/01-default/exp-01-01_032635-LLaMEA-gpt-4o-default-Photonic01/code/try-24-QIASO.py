import numpy as np

class QIASO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = min(50, budget)
        self.positions = None
        self.velocities = None
        self.best_personal_positions = None
        self.best_personal_values = None
        self.best_global_position = None
        self.best_global_value = np.inf
        self.w = 0.5  # Inertia weight
        self.c1 = 1.5  # Cognitive (personal) coefficient
        self.c2 = 1.5  # Social (global) coefficient
        self.q_factor = 0.1  # Quantum factor for position update

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.swarm_size, self.dim))
        self.best_personal_positions = self.positions.copy()
        self.best_personal_values = np.full(self.swarm_size, np.inf)

    def quantum_update(self, position, best):
        delta = np.random.normal(0, self.q_factor, self.dim)
        quantum_position = position + delta * (best - position)
        return quantum_position

    def adaptive_parameter_control(self, iteration, max_iterations):
        self.w = 0.4 + 0.5 * (1 - iteration / max_iterations)
        self.c1 = 2.0 - 1.5 * (iteration / max_iterations)
        self.c2 = 0.5 + 1.5 * (iteration / max_iterations)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0
        iteration = 0
        max_iterations = self.budget // self.swarm_size

        while evaluations < self.budget:
            self.adaptive_parameter_control(iteration, max_iterations)

            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                # Update velocity
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.best_personal_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.best_global_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component

                # Update position
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = self.quantum_update(self.positions[i], self.best_global_position)
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                # Evaluate new position
                current_value = func(self.positions[i])
                evaluations += 1

                # Update personal best
                if current_value < self.best_personal_values[i]:
                    self.best_personal_values[i] = current_value
                    self.best_personal_positions[i] = self.positions[i].copy()

                # Update global best
                if current_value < self.best_global_value:
                    self.best_global_value = current_value
                    self.best_global_position = self.positions[i].copy()

            iteration += 1

        return self.best_global_position, self.best_global_value