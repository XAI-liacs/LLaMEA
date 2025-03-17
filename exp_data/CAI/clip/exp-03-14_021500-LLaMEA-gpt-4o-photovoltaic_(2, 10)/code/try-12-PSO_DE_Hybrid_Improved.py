import numpy as np

class PSO_DE_Hybrid_Improved:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particle_count = int(max(10, 0.1 * self.budget))  # Dynamic particle count
        self.particles = np.random.rand(self.particle_count, self.dim)
        self.velocities = np.random.rand(self.particle_count, self.dim)
        self.personal_best = self.particles.copy()
        self.personal_best_scores = np.full(self.particle_count, np.inf)
        self.global_best = None
        self.global_best_score = np.inf

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        eval_count = 0
        
        # Adaptive parameters
        inertia_weight = 0.9
        cognitive_coeff = 2.0
        social_coeff = 2.0
        F = 0.5
        CR = 0.9

        # Initialize particles
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

            # Update velocities and positions (PSO)
            inertia_weight = 0.5 + 0.4 * (1 - eval_count / self.budget)  # Adaptive inertia weight
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                cognitive = cognitive_coeff * r1 * (self.personal_best[i] - self.particles[i])
                social = social_coeff * r2 * (self.global_best - self.particles[i])
                self.velocities[i] = inertia_weight * self.velocities[i] + cognitive + social
                self.particles[i] += self.velocities[i]
                self.particles[i] = np.clip(self.particles[i], lb, ub)

            # Differential Evolution (Mutation and Crossover)
            for i in range(self.particle_count):
                if eval_count >= self.budget:
                    break
                idxs = [idx for idx in range(self.particle_count) if idx != i]
                a, b, c = self.particles[np.random.choice(idxs, 3, replace=False)]
                mutant = np.clip(a + F * (b - c) * np.random.rand(), lb, ub)  # Enhanced mutation strategy
                trial = np.where(np.random.rand(self.dim) < CR, mutant, self.particles[i])
                
                score = func(trial)
                eval_count += 1
                if score < self.personal_best_scores[i]:
                    self.personal_best_scores[i] = score
                    self.personal_best[i] = trial
                    if score < self.global_best_score:
                        self.global_best_score = score
                        self.global_best = trial

        return self.global_best, self.global_best_score