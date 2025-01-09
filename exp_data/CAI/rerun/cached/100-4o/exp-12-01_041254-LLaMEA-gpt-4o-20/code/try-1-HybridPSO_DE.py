import numpy as np

class HybridPSO_DE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 20
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.population_size, self.dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf
        self.f_evals = 0

    def __call__(self, func):
        while self.f_evals < self.budget:
            for i in range(self.population_size):
                current_score = func(self.particles[i])
                self.f_evals += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.particles[i]
                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.particles[i]

            if self.f_evals >= self.budget:
                break

            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                inertia = 0.5
                cognitive = 1.5
                social = 1.5
                self.velocities[i] = (inertia * self.velocities[i] +
                                      cognitive * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      social * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], self.lower_bound, self.upper_bound)

            # Differential Evolution mutation and crossover
            for i in range(self.population_size):
                indices = list(range(self.population_size))
                indices.remove(i)
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.personal_best_positions[a] + 0.8 * (self.personal_best_positions[b] - self.personal_best_positions[c])
                mutant_vector = np.clip(mutant_vector, self.lower_bound, self.upper_bound)
                crossover_rate = 0.9
                trial_vector = np.copy(self.particles[i])
                for j in range(self.dim):
                    if np.random.rand() < crossover_rate:
                        trial_vector[j] = mutant_vector[j]
                trial_score = func(trial_vector)
                self.f_evals += 1
                if trial_score < func(self.particles[i]):
                    self.particles[i] = trial_vector

        return self.global_best_position, self.global_best_score