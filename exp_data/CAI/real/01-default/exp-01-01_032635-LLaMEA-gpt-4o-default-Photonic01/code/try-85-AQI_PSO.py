import numpy as np

class AQI_PSO:
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
        self.mutation_rate = 0.1  # Adaptive mutation rate
        self.history = []

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.random.uniform(-0.1, 0.1, (self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def adaptive_mutation(self, position):
        mutation_vector = np.random.normal(0, 1, self.dim) * self.mutation_rate
        lb, ub = self.bounds
        new_position = position + mutation_vector
        return np.clip(new_position, lb, ub)

    def dynamic_inertia_and_mutation_adjustment(self):
        if len(self.history) < 2:
            return
        improvement_rate = (self.history[-2] - self.history[-1]) / self.history[-2]
        max_diversity = np.sqrt(np.sum((self.bounds[1] - self.bounds[0])**2))
        diversity = np.std(self.positions, axis=0).mean()
        self.w = 0.4 + 0.5 * (diversity / max_diversity)
        self.mutation_rate = max(0.05, min(0.2, self.mutation_rate * (1 + improvement_rate)))

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
            self.dynamic_inertia_and_mutation_adjustment()

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                if np.random.rand() < self.mutation_rate:
                    self.positions[i] = self.adaptive_mutation(self.positions[i])

        return self.global_best_position, self.global_best_value