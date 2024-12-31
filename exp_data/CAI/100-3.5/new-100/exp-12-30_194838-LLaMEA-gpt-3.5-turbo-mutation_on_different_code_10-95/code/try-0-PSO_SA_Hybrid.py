import numpy as np

class PSO_SA_Hybrid:
    def __init__(self, budget=10000, dim=10, swarm_size=30, max_iter=100):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.max_iter = max_iter
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        def sa_step(curr_x, curr_f):
            T = 1.0
            alpha = 0.9
            while T > 1e-6:
                new_x = curr_x + np.random.normal(0, T)
                new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
                
                new_f = func(new_x)
                if new_f < curr_f or np.random.rand() < np.exp((curr_f - new_f) / T):
                    curr_x, curr_f = new_x, new_f
                T *= alpha
            return curr_x, curr_f
        
        swarm = np.random.uniform(func.bounds.lb, func.bounds.ub, size=(self.swarm_size, self.dim))
        swarm_f = np.array([func(x) for x in swarm])
        pbest = swarm.copy()
        pbest_f = swarm_f.copy()
        gbest_idx = np.argmin(swarm_f)
        gbest = swarm[gbest_idx].copy()
        
        for _ in range(self.max_iter):
            for i in range(self.swarm_size):
                new_x = swarm[i] + np.random.uniform() * (gbest - swarm[i]) + np.random.uniform() * (pbest[i] - swarm[i])
                new_x = np.clip(new_x, func.bounds.lb, func.bounds.ub)
                
                new_f = func(new_x)
                if new_f < swarm_f[i]:
                    swarm[i] = new_x
                    swarm_f[i] = new_f
                    if new_f < pbest_f[i]:
                        pbest[i] = new_x
                        pbest_f[i] = new_f
                        
                    if new_f < swarm_f[gbest_idx]:
                        gbest_idx = i
                        gbest = swarm[i]
                        
            gbest, swarm_f[gbest_idx] = sa_step(gbest, swarm_f[gbest_idx])
            if swarm_f[gbest_idx] < self.f_opt:
                self.f_opt = swarm_f[gbest_idx]
                self.x_opt = gbest
                
        return self.f_opt, self.x_opt