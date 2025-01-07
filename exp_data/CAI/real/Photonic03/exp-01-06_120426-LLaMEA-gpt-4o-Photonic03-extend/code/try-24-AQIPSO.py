import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.729
        # Adaptive cognitive and social coefficients
        self.cognitive_coeff_min = 0.5
        self.cognitive_coeff_max = 2.5
        self.social_coeff_min = 0.5
        self.social_coeff_max = 2.5
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evaluations = 0

    def quantum_superposition(self, bounds):
        return bounds.lb + (bounds.ub - bounds.lb) * np.random.rand(self.dim)

    def local_search(self, position, bounds, amplitude):
        # Dynamic perturbation amplitude
        perturbation = np.random.uniform(-amplitude, amplitude, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, bounds.lb, bounds.ub)

    def __call__(self, func):
        bounds = func.bounds
        while self.evaluations < self.budget:
            for i in range(self.population_size):
                fitness = func(self.positions[i])
                self.evaluations += 1
                
                if fitness < self.personal_best_values[i]:
                    self.personal_best_values[i] = fitness
                    self.personal_best_positions[i] = self.positions[i]

                if fitness < self.global_best_value:
                    self.global_best_value = fitness
                    self.global_best_position = self.positions[i]

            for i in range(self.population_size):
                r1, r2 = np.random.rand(2)
                # Adaptive coefficients based on current evaluations
                self.cognitive_coeff = self.cognitive_coeff_min + \
                    (self.cognitive_coeff_max - self.cognitive_coeff_min) * (self.evaluations / self.budget)
                self.social_coeff = self.social_coeff_max - \
                    (self.social_coeff_max - self.social_coeff_min) * (self.evaluations / self.budget)
                cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], bounds.lb, bounds.ub)

                if np.random.rand() < 0.1:
                    self.positions[i] = self.quantum_superposition(bounds)
                    
                if np.random.rand() < 0.1:
                    amplitude = 0.2 - 0.15 * (self.evaluations / self.budget)
                    self.positions[i] = self.local_search(self.positions[i], bounds, amplitude)

        return self.global_best_position, self.global_best_value