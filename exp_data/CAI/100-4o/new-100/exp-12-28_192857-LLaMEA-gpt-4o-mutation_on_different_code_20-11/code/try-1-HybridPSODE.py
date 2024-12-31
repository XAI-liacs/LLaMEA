import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.num_particles = 40
        self.particles = np.random.uniform(-5.0, 5.0, (self.num_particles, dim))
        self.velocities = np.random.uniform(-1.0, 1.0, (self.num_particles, dim))
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.num_particles, np.inf)
        self.global_best = None
        self.global_best_score = np.inf
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        func_evals = 0
        while func_evals < self.budget:
            for i in range(self.num_particles):
                f = func(self.particles[i])
                func_evals += 1
                if f < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = f
                    self.personal_best[i] = self.particles[i].copy()

                if f < self.global_best_score:
                    self.global_best_score = f
                    self.global_best = self.particles[i].copy()

                if func_evals >= self.budget:
                    break

            # Update velocities and positions
            inertia = 0.4 + 0.1 * (1 - func_evals / self.budget)  # Dynamic inertia
            cognitive = 1.2
            social = 1.8
            for i in range(self.num_particles):
                r1, r2 = np.random.rand(2, self.dim)
                self.velocities[i] = (inertia * self.velocities[i] +
                                      cognitive * r1 * (self.personal_best[i] - self.particles[i]) +
                                      social * r2 * (self.global_best - self.particles[i]))
                self.particles[i] = np.clip(self.particles[i] + self.velocities[i], -5.0, 5.0)

                # Enhanced DE mutation based on rank
                rank_weight = 1 - (self.personal_best_scores[i] / np.max(self.personal_best_scores))
                if np.random.rand() < 0.4 + 0.2 * rank_weight:  # Probability based on rank
                    a, b, c = np.random.choice(self.num_particles, 3, replace=False)
                    mutant = self.particles[a] + 0.6 * (self.particles[b] - self.particles[c])
                    mutant = np.clip(mutant, -5.0, 5.0)
                    if func(mutant) < func(self.particles[i]):
                        self.particles[i] = mutant
                        func_evals += 1

            if self.global_best_score < self.f_opt:
                self.f_opt = self.global_best_score
                self.x_opt = self.global_best.copy()

        return self.f_opt, self.x_opt