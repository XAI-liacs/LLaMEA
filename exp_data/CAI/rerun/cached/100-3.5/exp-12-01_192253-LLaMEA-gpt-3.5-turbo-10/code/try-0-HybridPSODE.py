import numpy as np

class HybridPSODE:
    def __init__(self, budget, dim, swarm_size=30, f=0.5, cr=0.9, w=0.5):
        self.budget = budget
        self.dim = dim
        self.swarm_size = swarm_size
        self.f = f
        self.cr = cr
        self.w = w

    def __call__(self, func):
        swarm = np.random.uniform(-5.0, 5.0, (self.swarm_size, self.dim))
        velocities = np.zeros((self.swarm_size, self.dim))
        best_solution = swarm[np.argmin([func(x) for x in swarm])]
        
        for _ in range(self.budget):
            for i in range(self.swarm_size):
                r1, r2, r3 = np.random.choice(self.swarm_size, 3, replace=False)
                v = velocities[i] + self.f * (swarm[r1] - swarm[i]) + self.cr * (swarm[r2] - swarm[r3])
                new_position = swarm[i] + v
                new_position = np.clip(new_position, -5.0, 5.0)
                
                if func(new_position) < func(swarm[i]):
                    swarm[i] = new_position
                    velocities[i] = v

                if func(swarm[i]) < func(best_solution):
                    best_solution = swarm[i]

        return best_solution