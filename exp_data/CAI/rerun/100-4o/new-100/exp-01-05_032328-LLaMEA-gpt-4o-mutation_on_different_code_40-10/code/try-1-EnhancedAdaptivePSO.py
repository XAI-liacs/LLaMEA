import numpy as np

class EnhancedAdaptivePSO:
    def __init__(self, budget=10000, dim=10, swarm_size=50):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        x = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_x = np.copy(x)
        personal_best_f = np.full(self.swarm_size, np.Inf)
        
        global_best_x = None
        global_best_f = np.Inf
        
        c1, c2 = 1.49618, 1.49618
        inertia_weight = 0.9
        inertia_weight_min = 0.4

        evaluations = 0
        while evaluations < self.budget:
            current_swarm_size = max(1, int(self.swarm_size * (1 - evaluations / self.budget)))
            indices = np.random.choice(self.swarm_size, current_swarm_size, replace=False)
            for i in indices:
                f = func(x[i])
                evaluations += 1
                if f < personal_best_f[i]:
                    personal_best_f[i] = f
                    personal_best_x[i] = x[i]
                if f < global_best_f:
                    global_best_f = f
                    global_best_x = x[i]
            
            r1, r2 = np.random.rand(self.swarm_size, self.dim), np.random.rand(self.swarm_size, self.dim)
            v = (inertia_weight * v
                 + c1 * r1 * (personal_best_x - x)
                 + c2 * r2 * (global_best_x - x))
            x += v
            x = np.clip(x, lb, ub)

            if evaluations < self.budget * 0.5:
                x = x + np.random.normal(0, 0.1, x.shape)
            
            inertia_weight = max(inertia_weight_min, inertia_weight * 0.99)

        self.f_opt, self.x_opt = global_best_f, global_best_x
        return self.f_opt, self.x_opt