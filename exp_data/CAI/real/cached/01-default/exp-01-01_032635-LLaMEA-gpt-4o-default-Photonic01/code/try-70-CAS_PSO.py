import numpy as np

class CAS_PSO:
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
        self.bounds = None
        self.exploration_boost = 0.05  # Exploration boost factor

    def initialize_swarm(self, lb, ub):
        self.positions = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        self.velocities = np.zeros((self.swarm_size, self.dim))
        self.personal_best_positions = self.positions.copy()
        self.personal_best_values = np.full(self.swarm_size, np.inf)
        self.bounds = (lb, ub)

    def chaotic_map(self, x):
        # Using a simple logistic map for chaotic sequence generation
        r = 3.9
        return r * x * (1 - x)

    def stochastic_tunneling(self, position, func_value, global_best_value):
        tunneling_factor = -np.log(1 + np.exp(-0.5 * (func_value - global_best_value)))
        perturbation = np.random.normal(0, tunneling_factor, self.dim)
        return position + perturbation

    def update_parameters(self, iteration, max_iterations):
        chaos = self.chaotic_map(np.random.rand())
        self.w = 0.4 + chaos * 0.5  # Inertia weight evolves chaotically
        self.c1 = 1.5 + chaos * 0.5  # Cognitive coefficient evolves chaotically
        self.c2 = 1.5 + chaos * 0.5  # Social coefficient evolves chaotically

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        self.initialize_swarm(lb, ub)
        evaluations = 0
        iteration = 0

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

            self.update_parameters(iteration, self.budget // self.swarm_size)

            for i in range(self.swarm_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_component = self.c1 * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.c2 * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = self.w * self.velocities[i] + cognitive_component + social_component
                self.positions[i] += self.velocities[i]

                # Stochastic tunneling to help escape local optima
                if np.random.rand() < self.exploration_boost:
                    self.positions[i] = self.stochastic_tunneling(self.positions[i], current_value, self.global_best_value)

                # Ensure positions remain within bounds
                self.positions[i] = np.clip(self.positions[i], lb, ub)

            iteration += 1

        return self.global_best_position, self.global_best_value