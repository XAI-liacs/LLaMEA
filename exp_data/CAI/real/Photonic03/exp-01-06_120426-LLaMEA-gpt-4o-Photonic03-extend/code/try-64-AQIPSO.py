import numpy as np

class AQIPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.inertia_weight = 0.729
        self.cognitive_coeff = 1.49445
        self.social_coeff = 1.49445
        self.positions = np.random.rand(self.population_size, dim)
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_values = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evaluations = 0

    def quantum_superposition(self, bounds):
        randomness_factor = np.random.normal(0, 0.5, self.dim)
        return bounds.lb + (bounds.ub - bounds.lb) * randomness_factor

    def local_search(self, position, bounds):
        perturbation_range = np.linspace(0.05, 0.3, self.dim) * (1 - self.evaluations/self.budget)
        perturbation = np.random.uniform(-perturbation_range, perturbation_range, self.dim)
        new_position = position + perturbation
        return np.clip(new_position, bounds.lb, bounds.ub)

    def update_coefficients(self):
        self.cognitive_coeff = 1.49445 * (1 - self.evaluations/self.budget)
        self.social_coeff = 1.49445 * (self.evaluations/self.budget)
        self.inertia_weight = 0.95 - 0.6 * (self.evaluations / self.budget)

    def __call__(self, func):
        bounds = func.bounds
        while self.evaluations < self.budget:
            self.update_coefficients()
            if self.evaluations % (self.budget // 10) == 0:
                self.population_size = max(10, self.population_size - 2)
            
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
                cognitive_component = self.cognitive_coeff * r1 * (self.personal_best_positions[i] - self.positions[i])
                social_component = self.social_coeff * r2 * (self.global_best_position - self.positions[i])
                self.velocities[i] = (self.inertia_weight * self.velocities[i] +
                                      cognitive_component + social_component)
                self.velocities[i] = np.clip(self.velocities[i], -1, 1)  # Velocity saturation
                self.positions[i] = self.positions[i] + self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], bounds.lb, bounds.ub)

                dynamic_prob = 0.25 + 0.1 * np.sin(np.pi * (1 - self.evaluations/self.budget))  # Sinusoidal adaptive probability
                if np.random.rand() < dynamic_prob:
                    self.positions[i] = self.quantum_superposition(bounds)
                    
                if np.random.rand() < 0.2:
                    self.positions[i] = self.local_search(self.positions[i], bounds)

        return self.global_best_position, self.global_best_value