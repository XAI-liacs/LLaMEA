import numpy as np

class EnhancedHEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, 
                                           (self.population_size, self.dim))
        self.velocities = np.zeros((self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf

    def __call__(self, func):
        evaluations = 0
        inertia_weight = 0.9  # Adjusted for more exploration
        cognitive_weight = 1.4
        social_weight = 1.4

        while evaluations < self.budget:
            for i in range(self.population_size):
                score = func(self.particles[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.particles[i]

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.particles[i]

                if evaluations >= self.budget:
                    break

            r1 = np.random.uniform(0, 1, (self.population_size, self.dim))
            r2 = np.random.uniform(0, 1, (self.population_size, self.dim))

            self.velocities = (inertia_weight * self.velocities +
                               cognitive_weight * r1 * (self.personal_best_positions - self.particles) +
                               social_weight * r2 * (self.global_best_position - self.particles))
            
            self.particles += self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Evolutionary strategy: dynamically mutate with adaptive noise
            top_half_indices = np.argsort(self.personal_best_scores)[:self.population_size // 2]
            mutation_strength = 0.1 * (1 - evaluations / self.budget)  # Adaptive mutation
            for i in range(self.population_size // 2, self.population_size):
                parent_index = np.random.choice(top_half_indices)
                self.particles[i] = self.personal_best_positions[parent_index] + \
                                    np.random.normal(0, mutation_strength, self.dim)
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

        return self.global_best_position, self.global_best_score