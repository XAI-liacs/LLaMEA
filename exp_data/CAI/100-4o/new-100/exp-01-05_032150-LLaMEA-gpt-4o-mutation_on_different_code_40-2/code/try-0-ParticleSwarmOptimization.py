import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f_opt = np.Inf
        self.x_opt = None
        self.lb = -5.0
        self.ub = 5.0
        
        # PSO parameters
        self.w_max = 0.9  # inertia weight
        self.w_min = 0.4
        self.c1 = 2.0     # cognitive coefficient
        self.c2 = 2.0     # social coefficient

    def __call__(self, func):
        # Initialize particle positions and velocities
        x = np.random.uniform(self.lb, self.ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-abs(self.ub-self.lb), abs(self.ub-self.lb), (self.swarm_size, self.dim))
        p_best = np.copy(x)
        p_best_f = np.full(self.swarm_size, np.Inf)
        
        # Evaluate initial positions
        for i in range(self.swarm_size):
            f = func(x[i])
            p_best_f[i] = f
            if f < self.f_opt:
                self.f_opt = f
                self.x_opt = x[i]

        iteration = 0
        while iteration < self.budget:
            # Dynamically update inertia weight
            w = self.w_max - ((self.w_max - self.w_min) * (iteration / self.budget))
            
            for i in range(self.swarm_size):
                # Update velocities
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v[i] = (w * v[i] 
                        + self.c1 * r1 * (p_best[i] - x[i]) 
                        + self.c2 * r2 * (self.x_opt - x[i]))
                
                # Clamp velocities to prevent exceeding bounds
                v[i] = np.clip(v[i], -abs(self.ub-self.lb), abs(self.ub-self.lb))
                
                # Update positions
                x[i] = x[i] + v[i]
                x[i] = np.clip(x[i], self.lb, self.ub)
                
                # Evaluate new positions
                f = func(x[i])
                iteration += 1
                if f < p_best_f[i]:
                    p_best_f[i] = f
                    p_best[i] = x[i]
                if f < self.f_opt:
                    self.f_opt = f
                    self.x_opt = x[i]
                if iteration >= self.budget:
                    break
            
        return self.f_opt, self.x_opt