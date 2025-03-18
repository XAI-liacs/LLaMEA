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

        for i in range(self.particle_count):
            self.particles[i] = lb + (ub - lb) * self.particles[i]

        while eval_count < self.budget:
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                
                score = func(self.particles[i])
                eval_count += 1

                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = self.particles[i].copy()

                if score < self.global_best_score:
                    self.global_best_score = score
                    self.global_best = self.particles[i].copy()

            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = 2.0 * (0.5 + 0.5 * np.sin(np.pi * eval_count / self.budget)) * r1 * (self.personal_best[i] - self.particles[i])  # Oscillating cognitive component
                social = 2.0 * (0.5 + 0.5 * np.cos(np.pi * eval_count / self.budget)) * r2 * (self.global_best - self.particles[i])  # Oscillating social component
                inertia_weight = 0.5 * np.exp(-3 * eval_count / self.budget) + 0.4  # Non-linear dynamic inertia
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive + social
                max_velocity = 0.1 * (ub - lb) + 0.05 * (ub - lb) * np.cos(np.pi * eval_count / self.budget)
                self.velocities[i] = np.clip(self.velocities[i], -max_velocity, max_velocity)
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.particle_count) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutation_scaling = 0.85 + 0.15 * np.sin(np.pi * eval_count / self.budget)  # Adaptive mutation scaling
                mutant = np.clip(a + mutation_scaling * (b - c), lb, ub)
                crossover_rate = 0.5 + 0.4 * (1 - eval_count / self.budget)
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