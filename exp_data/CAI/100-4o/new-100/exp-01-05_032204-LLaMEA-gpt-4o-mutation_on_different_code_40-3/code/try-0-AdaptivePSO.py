import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.population_size = 50
        self.w = 0.9
        self.c1 = 2.0
        self.c2 = 2.0
        self.lb = -5.0
        self.ub = 5.0
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        # Initialize particles
        x = np.random.uniform(self.lb, self.ub, (self.population_size, self.dim))
        v = np.random.uniform(-1, 1, (self.population_size, self.dim))
        p_best = x.copy()
        p_best_val = np.array([func(p) for p in p_best])
        
        g_best = p_best[np.argmin(p_best_val)]
        g_best_val = np.min(p_best_val)
        
        evals = self.population_size
        
        while evals < self.budget:
            for i in range(self.population_size):
                r1, r2 = np.random.rand(), np.random.rand()
                v[i] = (self.w * v[i] + 
                        self.c1 * r1 * (p_best[i] - x[i]) + 
                        self.c2 * r2 * (g_best - x[i]))
                x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], self.lb, self.ub)
                
                f_val = func(x[i])
                evals += 1
                
                # Update personal and global best
                if f_val < p_best_val[i]:
                    p_best[i] = x[i]
                    p_best_val[i] = f_val
                    if f_val < g_best_val:
                        g_best = x[i]
                        g_best_val = f_val
                        self.f_opt, self.x_opt = g_best_val, g_best
                
                # Adaptive inertia weight update
                self.w = 0.4 + 0.5 * (evals / self.budget)
                
                if evals >= self.budget:
                    break
        
        return self.f_opt, self.x_opt