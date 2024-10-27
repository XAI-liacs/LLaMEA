import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 30
        self.max_iters = budget // self.swarm_size
        self.lb = -5.0
        self.ub = 5.0

    def __call__(self, func):
        def initialize_swarm(dim, swarm_size, lb, ub):
            return np.random.uniform(lb, ub, (swarm_size, dim))

        def clip_position(p, lb, ub):
            return np.clip(p, lb, ub)

        def optimize(func, swarm):
            best_pos = None
            best_val = np.inf
            for i in range(len(swarm)):
                fitness = func(swarm[i])
                if fitness < best_val:
                    best_val = fitness
                    best_pos = swarm[i]
            return best_pos

        swarm = initialize_swarm(self.dim, self.swarm_size, self.lb, self.ub)
        for _ in range(self.max_iters):
            for i in range(self.swarm_size):
                p_best = optimize(func, swarm)
                r1, r2 = np.random.uniform(0, 1, (2, self.dim))
                v = swarm[i] + 0.5 * (swarm[i] - swarm[i]) + 0.5 * (p_best - swarm[i]) + 0.5 * (r1 - r2)
                swarm[i] = clip_position(v, self.lb, self.ub)
        return optimize(func, swarm)