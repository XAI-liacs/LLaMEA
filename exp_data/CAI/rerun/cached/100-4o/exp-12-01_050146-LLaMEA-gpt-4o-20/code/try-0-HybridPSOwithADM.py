import numpy as np

class HybridPSOwithADM:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50  # Number of particles
        self.inertia_weight = 0.7
        self.cognitive_component = 1.5
        self.social_component = 1.5
        self.differential_weight = 0.8
        self.crossover_rate = 0.9
        self.velocity = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.position = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.personal_best_position = np.copy(self.position)
        self.personal_best_value = np.full(self.population_size, float('inf'))
        self.global_best_position = None
        self.global_best_value = float('inf')
        self.evaluate_count = 0

    def __call__(self, func):
        np.random.seed(42)  # For reproducibility

        while self.evaluate_count < self.budget:
            # Evaluate current population
            for i in range(self.population_size):
                if self.evaluate_count < self.budget:
                    current_value = func(self.position[i])
                    self.evaluate_count += 1
                    if current_value < self.personal_best_value[i]:
                        self.personal_best_value[i] = current_value
                        self.personal_best_position[i] = self.position[i]
                    if current_value < self.global_best_value:
                        self.global_best_value = current_value
                        self.global_best_position = self.position[i]
            
            # Update velocity and position using PSO
            for i in range(self.population_size):
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                cognitive_velocity = self.cognitive_component * r1 * (self.personal_best_position[i] - self.position[i])
                social_velocity = self.social_component * r2 * (self.global_best_position - self.position[i])
                self.velocity[i] = (self.inertia_weight * self.velocity[i]
                                    + cognitive_velocity
                                    + social_velocity)
                self.position[i] += self.velocity[i]
                self.position[i] = np.clip(self.position[i], self.lower_bound, self.upper_bound)

            # Apply Adaptive Differential Mutation
            for i in range(self.population_size):
                if self.evaluate_count < self.budget:
                    a, b, c = np.random.choice(self.population_size, 3, replace=False)
                    mutant_vector = self.position[a] + self.differential_weight * (self.position[b] - self.position[c])
                    mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                    trial_vector = np.copy(self.position[i])
                    crossover = np.random.rand(self.dim) < self.crossover_rate
                    trial_vector[crossover] = mutant_vector[crossover]
                    trial_value = func(trial_vector)
                    self.evaluate_count += 1
                    if trial_value < self.personal_best_value[i]:
                        self.position[i] = trial_vector
                        self.personal_best_value[i] = trial_value
                        self.personal_best_position[i] = trial_vector
                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial_vector

        return self.global_best_position, self.global_best_value