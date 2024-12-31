import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.9, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        lb, ub = func.bounds.lb, func.bounds.ub
        swarm_pos = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        swarm_vel = np.random.uniform(-1, 1, (self.swarm_size, self.dim))
        personal_best_pos = np.copy(swarm_pos)
        personal_best_val = np.array([func(p) for p in swarm_pos])
        global_best_pos = personal_best_pos[np.argmin(personal_best_val)]
        global_best_val = np.min(personal_best_val)
        
        eval_count = self.swarm_size
        
        while eval_count < self.budget:
            self.inertia *= 0.99  # Nonlinear inertia decay
            for i in range(len(swarm_pos)):
                if eval_count < self.budget // 2:  # Adaptive swarm size
                    self.swarm_size = min(50, self.swarm_size + 1)
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                swarm_vel[i] = (self.inertia * swarm_vel[i] +
                                self.cognitive * r1 * (personal_best_pos[i] - swarm_pos[i]) +
                                self.social * r2 * (global_best_pos - swarm_pos[i]))
                swarm_pos[i] += swarm_vel[i]
                swarm_pos[i] = np.clip(swarm_pos[i], lb, ub)
                
                f = func(swarm_pos[i])
                eval_count += 1
                
                if f < personal_best_val[i]:
                    personal_best_val[i] = f
                    personal_best_pos[i] = swarm_pos[i]
                
                if f < global_best_val:
                    global_best_val = f
                    global_best_pos = swarm_pos[i]
                
                if eval_count >= self.budget:
                    break

        self.f_opt = global_best_val
        self.x_opt = global_best_pos
        return self.f_opt, self.x_opt