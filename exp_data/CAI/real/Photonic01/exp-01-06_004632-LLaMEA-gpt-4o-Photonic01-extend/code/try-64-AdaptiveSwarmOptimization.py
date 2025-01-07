import numpy as np

class AdaptiveSwarmOptimization:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = min(50, self.budget // 5)  # Set population size
        self.inertia_weight = 0.7  # Initial inertia weight
        self.cognitive_constant = 1.5  # Cognitive constant
        self.social_constant = 2.2  # Social constant (adjusted from 2.1 to 2.2)
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
                dynamic_factor = eval_count / self.budget  # Adaptive factor based on the budget usage
                inertia = (0.5 + 0.35 * dynamic_factor) * self.velocities[i]  # Dynamic inertia weight
                cognitive = (self.cognitive_constant + 0.6 * dynamic_factor) * np.random.rand(self.dim) * (self.local_best[i] - self.positions[i])  # Updated cognitive constant
                social = self.social_constant * dynamic_factor * np.random.rand(self.dim) * (self.global_best - self.positions[i])
                perturbation = np.random.standard_normal(self.dim) * 0.03  # Enhanced chaotic perturbation
                self.velocities[i] = inertia + cognitive + social + perturbation
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] += np.random.normal(0, 0.007, self.dim)  # Increased Gaussian mutation

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
                        self.global_best = self.positions[i] + np.random.normal(0, 0.0001, self.dim)  # Perturbation

                if eval_count >= self.budget:
                    break

        return self.global_best, self.global_best_value