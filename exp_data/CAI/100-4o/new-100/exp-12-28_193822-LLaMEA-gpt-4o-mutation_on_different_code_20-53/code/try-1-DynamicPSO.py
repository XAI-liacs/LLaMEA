import numpy as np

class DynamicPSO:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.w_max = 0.9
        self.w_min = 0.4
        self.c1 = 1.5
        self.c2 = 1.5
        self.f_opt = np.inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        p_best = x.copy()
        p_best_val = np.array([func(xi) for xi in x])
        
        l_best = x.copy()  # Initialize local best

        g_best_idx = np.argmin(p_best_val)
        g_best = p_best[g_best_idx]
        g_best_val = p_best_val[g_best_idx]

        eval_count = self.swarm_size

        while eval_count < self.budget:
            w = self.w_max - (self.w_max - self.w_min) * (eval_count / self.budget)
            r1, r2 = np.random.rand(2)

            for i in range(self.swarm_size):
                neighbors = [p_best[(i+j) % self.swarm_size] for j in range(-1, 2)]
                l_best[i] = min(neighbors, key=func)  # Find local best within neighborhood
            
            v = w * v + self.c1 * r1 * (p_best - x) + self.c2 * r2 * (l_best - x)
            x = np.clip(x + v, lb, ub)

            for i in range(self.swarm_size):
                f = func(x[i])
                eval_count += 1
                if f < p_best_val[i]:
                    p_best[i] = x[i]
                    p_best_val[i] = f
                    if f < g_best_val:
                        g_best = x[i]
                        g_best_val = f

                if eval_count >= self.budget:
                    break

        self.f_opt = g_best_val
        self.x_opt = g_best
        return self.f_opt, self.x_opt