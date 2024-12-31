import numpy as np

class ParticleSwarmOptimization:
    def __init__(self, budget=10000, dim=10, swarm_size=30, inertia=0.7, cognitive=1.5, social=1.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.inertia = inertia
        self.cognitive = cognitive
        self.social = social
        self.f_opt = np.Inf
        self.x_opt = None

    def __call__(self, func):
        np.random.seed(42)  # Ensures repeatability
        lb, ub = func.bounds.lb, func.bounds.ub
        
        # Initialize particles
        x = np.random.uniform(lb, ub, (self.swarm_size, self.dim))
        v = np.random.uniform(-abs(ub - lb), abs(ub - lb), (self.swarm_size, self.dim))
        pbest = x.copy()
        pbest_values = np.array([func(p) for p in pbest])
        
        # Find initial global best
        gbest_value = np.min(pbest_values)
        gbest = x[np.argmin(pbest_values)]

        for _ in range(self.budget // self.swarm_size):
            for i in range(self.swarm_size):
                # Update velocities and positions
                r1, r2 = np.random.rand(self.dim), np.random.rand(self.dim)
                v[i] = (self.inertia * v[i] +
                        self.cognitive * r1 * (pbest[i] - x[i]) +
                        self.social * r2 * (gbest - x[i]))
                x[i] += v[i]
                x[i] = np.clip(x[i], lb, ub)  # Ensure within bounds
                
                # Evaluate new solution
                f = func(x[i])
                
                # Update personal best
                if f < pbest_values[i]:
                    pbest[i] = x[i]
                    pbest_values[i] = f
                
                # Update global best
                if f < gbest_value:
                    gbest = x[i]
                    gbest_value = f
            
        self.f_opt = gbest_value
        self.x_opt = gbest
        return self.f_opt, self.x_opt