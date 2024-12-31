import numpy as np

class CooperativeSwarmOptimization:
    def __init__(self, budget=10000, dim=10):
        self.budget = budget
        self.dim = dim
        self.f_opt = np.Inf
        self.x_opt = None
        self.swarm_size = 20
        self.pos = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        self.vel = np.random.uniform(-0.5, 0.5, (self.swarm_size, self.dim))
        self.p_best = self.pos.copy()
        self.p_best_val = np.full(self.swarm_size, np.Inf)
        self.global_best = None
        self.global_best_val = np.Inf

    def __call__(self, func):
        for _ in range(self.budget // self.swarm_size):
            # Evaluate the fitness of each particle
            for i in range(self.swarm_size):
                f = func(self.pos[i])
                if f < self.p_best_val[i]:
                    self.p_best_val[i] = f
                    self.p_best[i] = self.pos[i]
                if f < self.global_best_val:
                    self.global_best_val = f
                    self.global_best = self.pos[i]

            # Update velocities and positions
            for i in range(self.swarm_size):
                r1, r2 = np.random.rand(2)
                self.vel[i] = (0.5 * self.vel[i]
                               + 1.5 * r1 * (self.p_best[i] - self.pos[i])
                               + 1.5 * r2 * (self.global_best - self.pos[i]))
                self.pos[i] += self.vel[i]
                self.pos[i] = np.clip(self.pos[i], -5.0, 5.0)
                
            # Differential evolution style exploration
            for i in range(self.swarm_size):
                idxs = [idx for idx in range(self.swarm_size) if idx != i]
                a, b, c = np.random.choice(idxs, 3, replace=False)
                mutant = np.clip(self.pos[a] + 0.8 * (self.pos[b] - self.pos[c]), -5.0, 5.0)
                trial = np.where(np.random.rand(self.dim) < 0.9, mutant, self.pos[i])
                f_trial = func(trial)
                if f_trial < self.p_best_val[i]:
                    self.p_best_val[i] = f_trial
                    self.p_best[i] = trial
                if f_trial < self.global_best_val:
                    self.global_best_val = f_trial
                    self.global_best = trial
        
        return self.global_best_val, self.global_best