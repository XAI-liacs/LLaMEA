import numpy as np

class ADPC_DQA_PSO:
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
        self.exploration_boost = 0.05  # Exploration boost factor

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
        self.w = 0.4 + 0.5 * (diversity / max_diversity)

    def adaptive_parameter_control(self, global_improvement, local_improvement):
        if global_improvement > 0:
            self.c1 = min(2.0, self.c1 + 0.1)
            self.c2 = max(1.0, self.c2 - 0.1)
        if local_improvement:
            self.c1 = max(1.0, self.c1 - 0.05)
            self.c2 = min(2.0, self.c2 + 0.05)

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0
        previous_best_value = np.inf

        while evaluations < self.budget:
            global_improvement = False
            local_improvements = [False] * self.swarm_size

            for i in range(self.swarm_size):
                if evaluations >= self.budget:
                    break

                current_value = func(self.positions[i])
                evaluations += 1

                if current_value < self.personal_best_values[i]:
                    self.personal_best_values[i] = current_value
                    self.personal_best_positions[i] = self.positions[i].copy()
                    local_improvements[i] = True

                if current_value < self.global_best_value:
                    self.global_best_value = current_value
                    self.global_best_position = self.positions[i].copy()
                    global_improvement = True

            self.dynamic_inertia_adjustment()
            self.adaptive_parameter_control(global_improvement, any(local_improvements))

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