import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, w=0.9, c1=1.5, c2=1.5):
        self.budget = budget
        self.dim = dim
        self.w = w
        self.c1 = c1
        self.c2 = c2
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.bounds = (-5.0, 5.0)
        self.velocity_clamp = 0.1 * (self.bounds[1] - self.bounds[0])

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_pos = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_vel = np.zeros((self.swarm_size, self.dim))
        personal_best_pos = np.copy(swarm_pos)
        personal_best_val = np.full(self.swarm_size, np.Inf)
        
        for iteration in range(self.budget // self.swarm_size):
            self.w = 0.9 - 0.5 * (iteration / (self.budget // self.swarm_size))
            for i in range(self.swarm_size):
                f = func(swarm_pos[i])
                if f < personal_best_val[i]:
                    personal_best_val[i] = f
                    personal_best_pos[i] = swarm_pos[i]

                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = swarm_pos[i]

            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                neighborhood_best_pos = personal_best_pos[
                    np.argmin([func(p) for p in personal_best_pos])
                ]    
                swarm_vel[i] = (self.w * swarm_vel[i] +
                                self.c1 * r1 * (personal_best_pos[i] - swarm_pos[i]) +
                                self.c2 * r2 * (neighborhood_best_pos - swarm_pos[i]))
                swarm_vel[i] = np.clip(swarm_vel[i], -self.velocity_clamp, self.velocity_clamp)
                swarm_pos[i] += swarm_vel[i]
                swarm_pos[i] = np.clip(swarm_pos[i], lb, ub)

        return self.f_opt, self.x_opt