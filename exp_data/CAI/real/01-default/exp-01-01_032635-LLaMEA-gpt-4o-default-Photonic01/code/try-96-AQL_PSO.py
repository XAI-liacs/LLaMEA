import numpy as np

class AQL_PSO:
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
        self.w = 0.9
        self.c1 = 1.5
        self.c2 = 1.5
        self.c3 = 1.2  # Increased cooperative coefficient
        self.adapt_rate = 0.2  # Increased adaptation rate
        self.history = []

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def adaptive_quantum_position_update(self, position, global_best):
        alpha = 0.1 + 0.9 * np.random.rand()  # Adaptive scaling factor
        beta = np.random.normal(0, 1, self.dim)
        delta = np.random.normal(0, 1, self.dim) * 0.05
        new_position = position + alpha * beta * (global_best - position) + delta
        lb, ub = self.bounds
        return np.clip(new_position, lb, ub)

    def dynamic_inertia_adjustment(self):
        diversity = np.std(self.positions, axis=0).mean()
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        self.w = 0.4 + 0.5 * (diversity / max_diversity)
        if len(self.history) > 1:
            self.w += 0.1 * (self.history[-1] - self.history[-2]) / max(self.history)

    def localized_mutation(self, lb, ub):
        mutation_chance = 0.1  # Probability to apply mutation
        for i in range(self.swarm_size):
            if np.random.rand() < mutation_chance:
                mutation_radius = 0.05 * (ub - lb)
                mutation = np.random.uniform(-mutation_radius, mutation_radius, self.dim)
                self.positions[i] = np.clip(self.positions[i] + mutation, lb, ub)

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
            self.dynamic_inertia_adjustment()
            self.localized_mutation(lb, ub)

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                r3 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                cooperative_component = self.c3 * r3 * (np.mean(self.personal_best_positions, axis=0) - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component + cooperative_component
                self.positions[i] += self.velocities[i]

                if np.random.rand() < self.adapt_rate:
                    self.positions[i] = self.adaptive_quantum_position_update(self.positions[i], self.global_best_position)

        return self.global_best_position, self.global_best_value