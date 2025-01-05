import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.particles = 30
        self.c1 = 2.0  # cognitive parameter
        self.c2 = 2.0  # social parameter
        self.w = 0.85  # inertia weight (changed from 0.8 to 0.85)
        self.F = 0.5   # differential weight
        self.CR = 0.9  # crossover probability
        self.positions = None
        self.velocities = None
        self.personal_best = None
        self.personal_best_value = None
        self.global_best = None
        self.global_best_value = np.inf
        
    def __call__(self, func):
        np.random.seed(42)  # For reproducibility
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize the swarm
        self.positions = np.random.uniform(lb, ub, (self.particles, self.dim))
        self.velocities = np.random.uniform(-1, 1, (self.particles, self.dim))
        self.personal_best = self.positions.copy()
        self.personal_best_value = np.array([func(pos) for pos in self.positions])
        
        # Update global best
        min_idx = np.argmin(self.personal_best_value)
        self.global_best = self.personal_best[min_idx].copy()
        self.global_best_value = self.personal_best_value[min_idx]

        eval_count = self.particles

        while eval_count < self.budget:
            for i in range(self.particles):
                # Particle Swarm Optimization update
                rp = np.random.uniform(0, 1, self.dim)
                rg = np.random.uniform(0, 1, self.dim)
                self.velocities[i] = (self.w * self.velocities[i] +
                                      self.c1 * rp * (self.personal_best[i] - self.positions[i]) +
                                      self.c2 * rg * (self.global_best - self.positions[i]))
                self.positions[i] += self.velocities[i]
                self.positions[i] = np.clip(self.positions[i], lb, ub)

                # Differential Evolution mutation and crossover
                a, b, c = np.random.choice(self.particles, 3, replace=False)
                mutant = np.clip(self.positions[a] + self.F * (self.positions[b] - self.positions[c]), lb, ub)
                cross_points = np.random.rand(self.dim) < self.CR
                trial = np.where(cross_points, mutant, self.positions[i])

                trial_value = func(trial)
                eval_count += 1

                if trial_value < self.personal_best_value[i]:
                    self.personal_best[i] = trial
                    self.personal_best_value[i] = trial_value

                    if trial_value < self.global_best_value:
                        self.global_best = trial
                        self.global_best_value = trial_value

                if eval_count >= self.budget:
                    break

        return self.global_best