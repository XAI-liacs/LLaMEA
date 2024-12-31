import numpy as np

class HybridPSODE:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = 0.9  # maximum inertia
        self.w_min = 0.4  # minimum inertia
        self.c1 = 1.5  # cognitive component
        self.c2 = 1.5  # social component
        self.f = 0.8  # DE scaling factor
        self.cr = 0.9  # DE crossover rate
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pos = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.swarm_size, self.dim))

        p_best_pos = pos.copy()
        p_best_vals = np.array([func(ind) for ind in pos])

        g_best_idx = np.argmin(p_best_vals)
        g_best_pos = pos[g_best_idx].copy()
        self.f_opt = p_best_vals[g_best_idx]
        self.x_opt = g_best_pos.copy()

        eval_count = self.swarm_size

        while eval_count < self.budget:
            w = self.w_max - ((self.w_max - self.w_min) * (eval_count / self.budget))  # Adaptive inertia
            for i in range(self.swarm_size):
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                vel[i] = (w * vel[i] +
                          self.c1 * r1 * (p_best_pos[i] - pos[i]) +
                          self.c2 * r2 * (g_best_pos - pos[i]))
                pos[i] = pos[i] + vel[i]

                # Enforce boundary constraints
                pos[i] = np.clip(pos[i], func.bounds.lb, func.bounds.ub)

                # Evaluate and update personal best
                f_val = func(pos[i])
                eval_count += 1

                if f_val < p_best_vals[i]:
                    p_best_vals[i] = f_val
                    p_best_pos[i] = pos[i].copy()

                # Update global best
                if f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = pos[i].copy()

                # Differential evolution mutation and crossover
                if np.random.rand() < self.cr:
                    idxs = list(range(self.swarm_size))
                    idxs.remove(i)
                    a, b, c = pos[np.random.choice(idxs, 3, replace=False)]
                    mutant = np.clip(a + self.f * (b - c), func.bounds.lb, func.bounds.ub)

                    cross_points = np.random.rand(self.dim) < self.cr
                    cross_points[np.random.randint(self.dim)] = True
                    trial = np.where(cross_points, mutant, pos[i])

                    f_trial = func(trial)
                    eval_count += 1

                    if f_trial < p_best_vals[i]:
                        p_best_vals[i] = f_trial
                        p_best_pos[i] = trial.copy()

                        if f_trial < self.f_opt:
                            self.f_opt = f_trial
                            self.x_opt = trial.copy()

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt