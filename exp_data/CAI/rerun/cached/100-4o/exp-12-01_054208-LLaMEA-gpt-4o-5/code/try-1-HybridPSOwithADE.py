import numpy as np

class HybridPSOwithADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = min(100, 10 * dim)
        self.inertia_weight_initial = 0.9  # Changed to time-varying
        self.inertia_weight_final = 0.4  # New parameter
        self.cognitive_coeff = 1.5
        self.social_coeff = 1.5
        self.differential_weight = 0.8
        self.crossover_prob = 0.9
        self.positions = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.positions)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.evaluations = 0

    def __call__(self, func):
        while self.evaluations < self.budget:
            # Evaluate population
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                score = func(self.positions[i])
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.positions[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.positions[i]

            # PSO Update
            current_inertia_weight = (self.inertia_weight_initial - (self.evaluations / self.budget) * (self.inertia_weight_initial - self.inertia_weight_final))  # Adjusted inertia weight
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)
            self.velocities = (current_inertia_weight * self.velocities +
                               self.cognitive_coeff * r1 * (self.personal_best_positions - self.positions) +
                               self.social_coeff * r2 * (self.global_best_position - self.positions))
            self.positions += self.velocities
            self.positions = np.clip(self.positions, self.lower_bound, self.upper_bound)

            # ADE Update
            for i in range(self.population_size):
                if self.evaluations >= self.budget:
                    break
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.personal_best_positions[a] + self.differential_weight * (self.personal_best_positions[b] - self.personal_best_positions[c])
                trial_vector = np.copy(self.positions[i])
                crossover_mask = np.random.rand(self.dim) < self.crossover_prob
                trial_vector[crossover_mask] = mutant_vector[crossover_mask]
                trial_vector = np.clip(trial_vector, self.lower_bound, self.upper_bound)
                score = func(trial_vector)
                self.evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.positions[i] = trial_vector
                    self.personal_best_scores[i] = score
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best_position = trial_vector

        return self.global_best_position