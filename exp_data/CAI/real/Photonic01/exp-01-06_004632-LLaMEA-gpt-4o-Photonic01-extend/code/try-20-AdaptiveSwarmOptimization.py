import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 5)  # Set population size
        self.inertia_weight = 0.7  # Initial inertia weight
        self.cognitive_constant = 1.5  # Cognitive constant
        self.social_constant = 1.5  # Social constant (increased from 1.5 to 1.7)
        self.global_best = None
        self.global_best_value = float('inf')
        self.positions = None
        self.velocities = None
        self.local_best = None
        self.local_best_values = None

    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub
        eval_count = 0

        # Initialize particles
        self.positions = np.random.uniform(lb, ub, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-abs(ub-lb), abs(ub-lb), (self.population_size, self.dim))
        self.local_best = np.copy(self.positions)
        self.local_best_values = np.full(self.population_size, float('inf'))

        # Evaluate initial positions
        for i in range(self.population_size):
            value = func(self.positions[i])
            eval_count += 1
            self.local_best_values[i] = value
            if value < self.global_best_value:
                self.global_best_value = value
                self.global_best = self.positions[i]

        while eval_count < self.budget:
            for i in range(self.population_size):
                # Update velocities and positions
                inertia = (0.5 + (0.5 * eval_count / self.budget)) * self.velocities[i]  # Dynamic inertia weight
                cognitive = self.cognitive_constant * np.random.rand(self.dim) * (self.local_best[i] - self.positions[i])
                social = self.social_constant * np.random.rand(self.dim) * (self.global_best - self.positions[i])
                self.velocities[i] = inertia + cognitive + social
                self.positions[i] = self.positions[i] + self.velocities[i]

                # Ensure particles are within bounds
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                # Evaluate new position
                value = func(self.positions[i])
                eval_count += 1
                if value < self.local_best_values[i]:
                    self.local_best_values[i] = value
                    self.local_best[i] = self.positions[i]
                    if value < self.global_best_value:
                        self.global_best_value = value
                        self.global_best = self.positions[i]

                if eval_count >= self.budget:
                    break

        return self.global_best, self.global_best_value