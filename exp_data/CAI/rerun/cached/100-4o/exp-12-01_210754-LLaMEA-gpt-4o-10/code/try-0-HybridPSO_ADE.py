import numpy as np
from scipy.spatial.distance import cdist

class HybridPSO_ADE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.lb = -5.0
        self.ub = 5.0
        self.population_size = min(max(20, dim * 5), budget // 10)
        self.inertia = 0.7
        self.cognitive = 1.5
        self.social = 1.5
        self.mutation_factor = 0.8
        self.crossover_rate = 0.7

    def __call__(self, func):
        num_evaluations = 0

        # Initialize particle positions and velocities
        pos = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        vel = np.zeros((self.population_size, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.array([func(ind) for ind in pos])
        num_evaluations += self.population_size

        gbest_idx = np.argmin(pbest_val)
        gbest_pos = pbest_pos[gbest_idx, :]
        gbest_val = pbest_val[gbest_idx]

        while num_evaluations < self.budget:
            # Particle Swarm Optimization Phase
            r1 = np.random.rand(self.population_size, self.dim)
            r2 = np.random.rand(self.population_size, self.dim)

            vel = (self.inertia * vel +
                   self.cognitive * r1 * (pbest_pos - pos) +
                   self.social * r2 * (gbest_pos - pos))
            pos = pos + vel
            pos = np.clip(pos, self.lb, self.ub)

            for i in range(self.population_size):
                value = func(pos[i])
                num_evaluations += 1
                if num_evaluations >= self.budget:
                    break
                if value < pbest_val[i]:
                    pbest_val[i] = value
                    pbest_pos[i, :] = pos[i, :]
                    if value < gbest_val:
                        gbest_val = value
                        gbest_pos = pos[i, :]

            # Adaptive Differential Evolution Phase
            if num_evaluations < self.budget:
                indices = np.arange(self.population_size)
                for i in range(self.population_size):
                    idxs = np.random.choice(indices[indices != i], 3, replace=False)
                    x0, x1, x2 = pos[idxs]
                    mutant = np.clip(x0 + self.mutation_factor * (x1 - x2), self.lb, self.ub)
                    cross_points = np.random.rand(self.dim) < self.crossover_rate
                    if not np.any(cross_points):
                        cross_points[np.random.randint(0, self.dim)] = True
                    trial = np.where(cross_points, mutant, pos[i])
                    trial_value = func(trial)
                    num_evaluations += 1
                    if trial_value < pbest_val[i]:
                        pbest_val[i] = trial_value
                        pbest_pos[i, :] = trial.copy()
                        if trial_value < gbest_val:
                            gbest_val = trial_value
                            gbest_pos = trial.copy()

        return gbest_pos, gbest_val