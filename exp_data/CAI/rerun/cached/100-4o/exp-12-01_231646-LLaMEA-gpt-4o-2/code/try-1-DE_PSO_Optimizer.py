import numpy as np

class DE_PSO_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.population_size = 20
        self.population = np.random.uniform(-5.0, 5.0, (self.population_size, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.population_size, dim))
        self.personal_best_positions = np.copy(self.population)
        self.personal_best_scores = np.inf * np.ones(self.population_size)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover rate
        self.w = 0.5  # PSO inertia weight
        self.c1 = 1.5  # PSO cognitive coefficient
        self.c2 = 1.5  # PSO social coefficient

    def __call__(self, func):
        evaluations = 0
        while evaluations < self.budget:
            # Differential Evolution step
            for i in range(self.population_size):
                indices = list(range(0, i)) + list(range(i + 1, self.population_size))
                a, b, c = np.random.choice(indices, 3, replace=False)
                mutant_vector = self.population[a] + self.f * (self.population[b] - self.population[c])
                trial_vector = np.copy(self.population[i])
                j_rand = np.random.randint(self.dim)
                for j in range(self.dim):
                    if np.random.rand() < self.cr or j == j_rand:
                        trial_vector[j] = mutant_vector[j]
                
                # Evaluate trial vector
                trial_score = func(trial_vector)
                evaluations += 1
                if evaluations >= self.budget:
                    break
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial_vector
                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial_vector
            
            if evaluations >= self.budget:
                break

            # Particle Swarm Optimization step
            for i in range(self.population_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                self.w = 0.4 + 0.5 * ((self.budget - evaluations) / self.budget)  # Adaptive inertia weight
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.population[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.population[i]))
                self.population[i] += self.velocities[i]
                
                # Ensure boundary constraints
                self.population[i] = np.clip(self.population[i], -5.0, 5.0)
                
                # Evaluate the new position
                score = func(self.population[i])
                evaluations += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best_positions[i] = self.population[i]
                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best_position = self.population[i]
                
                if evaluations >= self.budget:
                    break

        return self.global_best_position