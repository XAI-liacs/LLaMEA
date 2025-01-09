import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.pop_size = 40
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.vel_bound = (self.upper_bound - self.lower_bound) / 2.0
        self.inertia_weight = 0.729
        self.cognitive_coeff = 1.49445
        self.social_coeff = 1.49445
        self.mutation_factor = 0.8
        self.crossover_rate = 0.9
        self.population = np.random.uniform(self.lower_bound, self.upper_bound, (self.pop_size, self.dim))
        self.velocity = np.random.uniform(-self.vel_bound, self.vel_bound, (self.pop_size, self.dim))
        self.personal_best_position = np.copy(self.population)
        self.personal_best_value = np.full(self.pop_size, np.inf)
        self.global_best_position = None
        self.global_best_value = np.inf
        self.evaluation_count = 0

    def __call__(self, func):
        while self.evaluation_count < self.budget:
            for i, particle in enumerate(self.population):
                fitness_value = func(particle)
                self.evaluation_count += 1

                if fitness_value < self.personal_best_value[i]:
                    self.personal_best_value[i] = fitness_value
                    self.personal_best_position[i] = particle

                if fitness_value < self.global_best_value:
                    self.global_best_value = fitness_value
                    self.global_best_position = particle

            for i, particle in enumerate(self.population):
                r_p = np.random.rand(self.dim)
                r_g = np.random.rand(self.dim)
                self.velocity[i] = (self.inertia_weight * self.velocity[i] +
                                    self.cognitive_coeff * r_p * (self.personal_best_position[i] - particle) +
                                    self.social_coeff * r_g * (self.global_best_position - particle))
                self.velocity[i] = np.clip(self.velocity[i], -self.vel_bound, self.vel_bound)
                self.population[i] = particle + self.velocity[i]
                self.population[i] = np.clip(self.population[i], self.lower_bound, self.upper_bound)

            if self.evaluation_count >= self.budget:
                break
            
            trial_population = np.copy(self.population)
            for i in range(self.pop_size):
                if self.evaluation_count >= self.budget:
                    break
                indices = list(range(self.pop_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = np.clip(self.population[a] +
                                        self.mutation_factor * (self.personal_best_position[b] - self.personal_best_position[c]),
                                        self.lower_bound, self.upper_bound)
                crossover_mask = np.random.rand(self.dim) < self.crossover_rate
                if not np.any(crossover_mask):
                    crossover_mask[np.random.randint(0, self.dim)] = True
                trial_vector = np.where(crossover_mask, mutant_vector, self.population[i])
                trial_value = func(trial_vector)
                self.evaluation_count += 1
                
                if trial_value < self.personal_best_value[i]:
                    self.population[i] = trial_vector
                    self.personal_best_value[i] = trial_value
                    self.personal_best_position[i] = trial_vector

                    if trial_value < self.global_best_value:
                        self.global_best_value = trial_value
                        self.global_best_position = trial_vector
            
            # Adaptive parameters and elitism
            self.inertia_weight = 0.9 * (1 - self.evaluation_count / self.budget)
            elite_idx = np.argmin(self.personal_best_value)
            self.population[elite_idx] = self.global_best_position

        return self.global_best_value, self.global_best_position