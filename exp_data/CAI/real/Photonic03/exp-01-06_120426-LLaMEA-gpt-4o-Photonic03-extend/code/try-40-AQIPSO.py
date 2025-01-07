import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.729  # Inertia weight to control exploration and exploitation
        self.cognitive_coeff = 1.49445  # Cognitive coefficient
        self.social_coeff = 1.49445  # Social coefficient
        self.positions = np.random.rand(self.population_size, dim)  # Initial positions
        self.velocities = np.zeros((self.population_size, dim))  # Initial velocities
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evaluations = 0

    def quantum_superposition(self, bounds):
        # Generate a new position using quantum superposition-like strategy
        return bounds.lb + (bounds.ub - bounds.lb) * np.random.rand(self.dim)

    def local_search(self, position, bounds):
        # Implementing a more aggressive local search by perturbing the position
        perturbation_range = np.linspace(0.05, 0.3, self.dim)  # Dynamic perturbation range
        perturbation = np.random.uniform(-perturbation_range, perturbation_range, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                # Evaluate the fitness of each particle
                fitness = func(self.positions[i])
                self.evaluations += 1
                
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i]

            # Update velocities and positions
            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                # Modify inertia weight based on evaluations
                self.inertia_weight = 0.9 - (0.5 * self.evaluations / self.budget)
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                self.positions[i] = self.positions[i] + self.velocities[i]

                # Ensure positions are within bounds
                self.positions[i] = np.clip(self.positions[i], bounds.lb, bounds.ub)

                # Apply quantum-inspired update and local search occasionally
                if np.random.rand() < 0.1:
                    self.positions[i] = self.quantum_superposition(bounds)
                    
                if np.random.rand() < 0.1:
                    self.positions[i] = self.local_search(self.positions[i], bounds)

        return self.global_best_position, self.global_best_value