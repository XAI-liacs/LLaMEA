import numpy as np

class PSO_ADP:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.c1 = 1.5  # cognitive coefficient
        self.c2 = 1.5  # social coefficient
        self.w = 0.5   # inertia weight
        self.pop_size = 50
        self.f_opt = np.Inf
        self.x_opt = None
    
    def __call__(self, func):
        lb = func.bounds.lb
        ub = func.bounds.ub

        # Initialize particles
        x = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        v = np.random.uniform(-1, 1, (self.pop_size, self.dim))
        p_best = x.copy()
        p_best_val = np.array([func(indiv) for indiv in p_best])
        self.f_opt, g_best_idx = np.min(p_best_val), np.argmin(p_best_val)
        self.x_opt = p_best[g_best_idx]

        eval_count = self.pop_size

        while eval_count < self.budget:
            for i in range(self.pop_size):
                # Update velocity and position
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v[i] = (self.w * v[i] +
                        self.c1 * r1 * (p_best[i] - x[i]) +
                        self.c2 * r2 * (self.x_opt - x[i]))
                x[i] = np.clip(x[i] + v[i], lb, ub)

                # Adaptive Differential Perturbation (ADP)
                perturbed_x = x[i] + 0.1 * np.random.randn(self.dim)
                perturbed_x = np.clip(perturbed_x, lb, ub)
                
                f_val = func(x[i])
                perturbed_f_val = func(perturbed_x)
                eval_count += 2

                # Update personal best and global best
                if f_val < p_best_val[i]:
                    p_best[i] = x[i]
                    p_best_val[i] = f_val
                if perturbed_f_val < p_best_val[i]:
                    p_best[i] = perturbed_x
                    p_best_val[i] = perturbed_f_val
                
                if perturbed_f_val < self.f_opt:
                    self.f_opt = perturbed_f_val
                    self.x_opt = perturbed_x
                elif f_val < self.f_opt:
                    self.f_opt = f_val
                    self.x_opt = x[i]

                if eval_count >= self.budget:
                    break

        return self.f_opt, self.x_opt