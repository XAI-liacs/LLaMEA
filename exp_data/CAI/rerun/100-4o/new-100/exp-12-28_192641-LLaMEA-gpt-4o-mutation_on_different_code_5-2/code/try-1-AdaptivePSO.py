import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.pop_size = 40
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 2.5  # Adjusted from 2.0 to 2.5
        self.c2 = 2.0

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        pop = np.random.uniform(lb, ub, (self.pop_size, self.dim))
        vel = np.random.uniform(-(ub - lb) * 0.5, (ub - lb) * 0.5, (self.pop_size, self.dim))  # Adjusted velocity bounds
        p_best = pop.copy()
        p_best_val = np.array([func(ind) for ind in pop])
        g_best_idx = np.argmin(p_best_val)
        g_best = pop[g_best_idx].copy()
        g_best_val = p_best_val[g_best_idx]
        
        evaluations = self.pop_size
        while evaluations < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (evaluations / self.budget)
            r1, r2 = np.random.rand(self.pop_size, self.dim), np.random.rand(self.pop_size, self.dim)
            vel = (w * vel +
                   self.c1 * r1 * (p_best - pop) +
                   self.c2 * r2 * (g_best - pop))
            pop = np.clip(pop + vel, lb, ub)
            new_vals = np.array([func(ind) for ind in pop])
            evaluations += self.pop_size
            
            for i in range(self.pop_size):
                if new_vals[i] < p_best_val[i]:
                    p_best_val[i] = new_vals[i]
                    p_best[i] = pop[i]
            
            g_best_idx = np.argmin(p_best_val)
            if p_best_val[g_best_idx] < g_best_val:
                g_best_val = p_best_val[g_best_idx]
                g_best = p_best[g_best_idx].copy()
        
        self.f_opt = g_best_val
        self.x_opt = g_best
        return self.f_opt, self.x_opt