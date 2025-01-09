import numpy as np

class PSO_DE_Optimizer:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lower_bound = -5.0
        self.upper_bound = 5.0
        self.n_particles = 50  # Number of particles
        self.particles = np.random.uniform(self.lower_bound, self.upper_bound, (self.n_particles, self.dim))
        self.velocities = np.zeros((self.n_particles, self.dim))
        self.personal_best_positions = self.particles.copy()
        self.personal_best_scores = np.full(self.n_particles, np.inf)
        self.global_best_position = np.zeros(self.dim)
        self.global_best_score = np.inf
        self.w = 0.7  # Inertia weight
        self.c1 = 1.5  # Cognitive coefficient
        self.c2 = 1.5  # Social coefficient
        self.F = 0.5  # Differential evolution scale factor
        self.CR = 0.9  # Differential evolution crossover probability

    def __call__(self, func):
        evals = 0
        while evals < self.budget:
            for i in range(self.n_particles):
                current_score = func(self.particles[i])
                evals += 1
                if current_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = current_score
                    self.personal_best_positions[i] = self.particles[i]

                if current_score < self.global_best_score:
                    self.global_best_score = current_score
                    self.global_best_position = self.particles[i]

                if evals >= self.budget:
                    break

            # Update particle velocities and positions based on PSO
            r1 = np.random.rand(self.n_particles, self.dim)
            r2 = np.random.rand(self.n_particles, self.dim)
            self.velocities = (
                self.w * self.velocities
                + self.c1 * r1 * (self.personal_best_positions - self.particles)
                + self.c2 * r2 * (self.global_best_position - self.particles)
            )
            self.particles = self.particles + self.velocities
            self.particles = np.clip(self.particles, self.lower_bound, self.upper_bound)

            # Apply Differential Evolution on a subset of particles
            for i in range(self.n_particles):
                indices = np.random.choice(self.n_particles, 3, replace=False)
                x0, x1, x2 = self.particles[indices]
                mutant = x0 + self.F * (x1 - x2)
                mutant = np.clip(mutant, self.lower_bound, self.upper_bound)

                trial = np.where(np.random.rand(self.dim) < self.CR, mutant, self.particles[i])
                trial_score = func(trial)
                evals += 1

                if trial_score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = trial_score
                    self.personal_best_positions[i] = trial

                if trial_score < self.global_best_score:
                    self.global_best_score = trial_score
                    self.global_best_position = trial

                if evals >= self.budget:
                    break

        return self.global_best_position, self.global_best_score