import numpy as np

class AdaptivePSO:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.swarm_size = 50
        self.c1 = 2.0
        self.c2 = 2.0
        self.w_max = 0.9
        self.w_min = 0.4
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        pos = np.random.uniform(func.bounds.lb, func.bounds.ub, (self.swarm_size, self.dim))
        vel = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        pbest_pos = np.copy(pos)
        pbest_val = np.full(self.swarm_size, np.inf)
        gbest_pos = None
        gbest_val = np.inf

        iteration = 0
        while iteration < self.budget:
            for i in range(self.swarm_size):
                f = func(pos[i])
                if f < pbest_val[i]:
                    pbest_val[i] = f
                    pbest_pos[i] = pos[i]
                
                if f < gbest_val:
                    gbest_val = f
                    gbest_pos = pos[i]

            if iteration % 10 == 0:
                self.local_search(func, gbest_pos)

            w = self.w_max - (iteration / self.budget) * (self.w_max - self.w_min)
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(), np.random.rand()
                vel[i] = (w * vel[i] 
                          + self.c1 * 1.1 * r1 * (pbest_pos[i] - pos[i])  # Adjusted line
                          + self.c2 * r2 * (gbest_pos - pos[i]))
                pos[i] = pos[i] + vel[i]
                pos[i] = np.clip(pos[i], func.bounds.lb, func.bounds.ub)

            iteration += self.swarm_size

        return gbest_val, gbest_pos

    def local_search(self, func, best_pos):
        neighborhood = np.random.normal(best_pos, 0.2, (5, self.dim))  # Adjusted line
        for i in range(neighborhood.shape[0]):
            f = func(neighborhood[i])
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = neighborhood[i]