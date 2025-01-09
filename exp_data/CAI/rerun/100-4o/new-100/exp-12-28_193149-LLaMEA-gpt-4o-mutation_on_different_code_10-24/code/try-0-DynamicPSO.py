import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.num_particles = min(100, budget // dim)
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.0
        self.c2 = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        # Initialize particles
        x = np.random.uniform(lb, ub, (self.num_particles, self.dim))
        v = np.random.uniform(-1, 1, (self.num_particles, self.dim))
        personal_best_x = np.copy(x)
        personal_best_f = np.array([func(xi) for xi in x])
        
        # Initialize global best
        global_best_idx = np.argmin(personal_best_f)
        self.f_opt = personal_best_f[global_best_idx]
        self.x_opt = personal_best_x[global_best_idx]

        function_evals = self.num_particles

        while function_evals < self.budget:
            # Update particle velocities and positions
            inertia_weight = self.w_max - ((self.w_max - self.w_min) * function_evals / self.budget)
            r1, r2 = np.random.rand(self.num_particles, self.dim), np.random.rand(self.num_particles, self.dim)
            v = (inertia_weight * v
                 + self.c1 * r1 * (personal_best_x - x)
                 + self.c2 * r2 * (self.x_opt - x))
            
            # Adaptive velocity clamping
            v_max = (ub - lb) / 2
            v = np.clip(v, -v_max, v_max)

            x = x + v
            x = np.clip(x, lb, ub)

            # Evaluate new solutions
            f_values = np.array([func(xi) for xi in x])
            function_evals += self.num_particles

            # Update personal and global bests
            improved = f_values < personal_best_f
            personal_best_x[improved] = x[improved]
            personal_best_f[improved] = f_values[improved]

            if np.min(f_values) < self.f_opt:
                global_best_idx = np.argmin(f_values)
                self.f_opt = f_values[global_best_idx]
                self.x_opt = x[global_best_idx]

        return self.f_opt, self.x_opt