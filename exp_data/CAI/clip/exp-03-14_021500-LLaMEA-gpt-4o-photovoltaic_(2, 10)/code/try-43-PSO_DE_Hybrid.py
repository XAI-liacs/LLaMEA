import numpy as np

class PSO_DE_Hybrid:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = 10
        self.particles = np.random.rand(self.particle_count, self.dim)
        self.velocities = np.random.rand(self.particle_count, self.dim)
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.particle_count, np.inf)
        self.global_best = None
        self.global_best_score = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0

        # Initialize particles
        for i in range(self.particle_count):
            self.particles[i] = lb + (ub - lb) * self.particles[i]

        while eval_count < self.budget:
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                
                # Evaluate current particle
                score = func(self.particles[i])
                eval_count += 1

                # Update personal and global bests
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            # Update velocities and positions (PSO)
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = 2.0 * r1 * (self.personal_best[i] - self.particles[i])
                social = 2.0 * r2 * (self.global_best - self.particles[i])
                phase_factor = 0.5 + 0.5 * (1 - eval_count / self.budget)  # New phase factor
                inertia_weight = (0.9 - 0.5 * (self.personal_best_scores[i] / self.global_best_score)) * phase_factor  # Adjusted inertia
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

            # Differential Evolution (Mutation and Crossover)
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.particle_count) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                adaptive_factor = 0.5 + 0.3 * (1 - eval_count / self.budget)  # Adaptive scaling factor
                mutant = np.clip(a + adaptive_factor * (b - c), lb, ub)  # Updated scaling factor line
                crossover_rate = 0.5 + 0.4 * (1 - eval_count / self.budget)  # Adaptive crossover rate
                trial = np.where(np.random.rand(self.dim) < crossover_rate, mutant, self.particles[i])
                
                score = func(trial)
                eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial

        return self.global_best, self.global_best_score