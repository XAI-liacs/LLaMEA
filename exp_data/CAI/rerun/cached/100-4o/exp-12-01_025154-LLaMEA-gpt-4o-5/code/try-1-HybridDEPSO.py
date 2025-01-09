import numpy as np

class HybridDEPSO:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.population_size = 50
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.population_size, dim))
        self.velocities = np.zeros((self.population_size, dim))
        self.personal_best_positions = np.copy(self.particles)
        self.personal_best_scores = np.full(self.population_size, np.inf)
        self.global_best_position = None
        self.global_best_score = np.inf
        self.w = 0.9  # Inertia weight, increased for better exploration initially
        self.c1 = 1.5 # Cognitive coefficient
        self.c2 = 1.5 # Social coefficient
        self.de_f = 0.8  # Differential Evolution scaling factor
        self.de_cr = 0.9 # Crossover probability

    def __call__(self, func):
        evals = 0
        
        while evals < self.budget:
            # Differential Evolution step
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                indices = np.random.choice(np.delete(np.arange(self.population_size), i), 3, replace=False)
                a, b, c = self.particles[indices]
                mutant_vector = np.clip(a + self.de_f * (b - c), self.lower_bound, self.upper_bound)
                trial_vector = np.copy(self.particles[i])
                
                for j in range(self.dim):
                    if np.random.rand() < self.de_cr:
                        trial_vector[j] = mutant_vector[j]
                
                trial_score = func(trial_vector)
                evals += 1
                
                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_positions[i] = trial_vector
                    self.personal_best_scores[i] = trial_score

            # Update global best
            min_personal_best_score = np.min(self.personal_best_scores)
            if min_personal_best_score < self.global_best_score:
                self.global_best_score = min_personal_best_score
                self.global_best_position = self.personal_best_positions[np.argmin(self.personal_best_scores)]

            # Particle Swarm Optimization step, using personal and global bests
            self.w = 0.9 - 0.5 * (evals / self.budget)  # Adaptive inertia weight
            for i in range(self.population_size):
                if evals >= self.budget:
                    break
                r1 = np.random.rand(self.dim)
                r2 = np.random.rand(self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * r1 * (self.personal_best_positions[i] - self.particles[i]) +
                                      self.c2 * r2 * (self.global_best_position - self.particles[i]))
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], self.lower_bound, self.upper_bound)

                particle_score = func(self.particles[i])
                evals += 1
                
                if particle_score < self.personal_best_scores[i]:
                    self.personal_best_positions[i] = self.particles[i]
                    self.personal_best_scores[i] = particle_score

        return self.global_best_position, self.global_best_score