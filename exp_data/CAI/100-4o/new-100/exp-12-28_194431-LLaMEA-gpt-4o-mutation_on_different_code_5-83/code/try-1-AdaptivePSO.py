import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles' positions and velocities
        x = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        
        # Initialize personal bests and global best
        p_best = x.copy()
        p_best_val = np.array([func(xi) for xi in x])
        g_best = x[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)
        
        # Update global best
        if g_best_val < self.f_opt:
            self.f_opt = g_best_val
            self.x_opt = g_best
        
        evaluations = self.swarm_size
        w_max, w_min = 0.9, 0.4
        c1_init, c2_init = 2.05, 2.05
        chi = 0.729  # Constriction factor

        while evaluations < self.budget:
            # Adaptive inertia weight
            w = w_max - ((w_max - w_min) * evaluations / self.budget)
            c1 = c1_init * (1 - evaluations / self.budget)  # Dynamic cognitive coefficient
            c2 = c2_init * (evaluations / self.budget)      # Dynamic social coefficient
            
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                # Update velocity
                v[i] = chi * (w * v[i] + c1 * r1 * (p_best[i] - x[i]) + c2 * r2 * (g_best - x[i]))
                # Update position
                x[i] = x[i] + v[i]
                # Ensure particles stay within bounds
                x[i] = np.clip(x[i], func.bounds.lb, func.bounds.ub)
                
                # Evaluate new position
                f = func(x[i])
                evaluations += 1
                
                # Update personal and global bests
                if f < p_best_val[i]:
                    p_best[i] = x[i].copy()
                    p_best_val[i] = f
                    if f < g_best_val:
                        g_best = x[i].copy()
                        g_best_val = f
                        if g_best_val < self.f_opt:
                            self.f_opt = g_best_val
                            self.x_opt = g_best
            
            if evaluations >= self.budget:
                break

        return self.f_opt, self.x_opt